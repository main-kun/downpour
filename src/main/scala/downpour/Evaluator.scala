package downpour

import akka.actor.{Actor, ActorLogging, ActorRef}
import breeze.linalg.argmax
import breeze.numerics.sigmoid
import downpour.Evaluator.EvaluateModel
import downpour.Types._

import scala.concurrent.duration._
import scala.concurrent.Await
import akka.pattern.ask
import akka.util.Timeout
import downpour.ParameterServer.FetchParameters

object Evaluator {
  case class EvaluateModel()
}

class Evaluator(testData: TrainingTupleVector,
                parameterServer: ActorRef) extends Actor with ActorLogging {

  def feedForward(input: TrainingExample,
                  biases: BiasSeq,
                  weights: WeightSeq): TrainingExample = {
    biases.zip(weights).foldLeft(input) {
      case (acc, (b, w)) => sigmoid(w * acc + b)
    }
  }

  def evaluate(biases: BiasSeq, weights: WeightSeq): Int = {
    val testResults = testData.map {
      case (x, y) =>
        val output = feedForward(x, biases, weights)
        val maxOutput = argmax(output)
        val maxTarget = argmax(y)
        if (maxTarget == maxOutput) 1 else 0
    }
    testResults.sum
  }

  def receive = {
    case EvaluateModel =>
      implicit val timeout = Timeout(5 seconds)
      val parameterFuture = parameterServer ? FetchParameters
      val parameterTuple = Await.result(parameterFuture, timeout.duration).asInstanceOf[ParameterTuple]
      val (biases, weights) = parameterTuple
      val result = evaluate(biases, weights)
      log.info(s"EVALUATION $result / ${testData.length}")
  }

}
