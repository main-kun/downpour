package downpour

import java.io.File

import akka.actor.{Actor, ActorLogging, ActorRef}
import breeze.linalg.argmax
import breeze.numerics.sigmoid
import downpour.Evaluator.{EvaluateModel, EvaluatorDone, StartTimer}
import downpour.Types._

import scala.concurrent.duration._
import scala.concurrent.Await
import akka.pattern.ask
import akka.util.Timeout
import downpour.Master.MasterDone
import downpour.ParameterServer.FetchParameters

/**
  * Evaluator - Actor evaluating the model
  *
  * @constructor Create Evaluator instance
  *
  * @param testData Vector of testing objects
  * @param parameterServer Actor ref of ParameterServer
  * @parallelFactor Number of Replicas in system
  */

object Evaluator {
  case class EvaluateModel()
  case class StartTimer()
  case class EvaluatorDone()
}

class Evaluator(testData: TrainingTupleVector,
                parameterServer: ActorRef,
                parallelFactor: Int) extends Actor with ActorLogging {

  var startTime: Long = 0
  var results =  Seq.empty[(Int, Long)]
  var counter: Int = 0
  var doneCounter: Int = 0

  def feedForward(input: TrainingExample,
                  biases: BiasSeq,
                  weights: WeightSeq): TrainingExample = {
    biases.zip(weights).foldLeft(input) {
      case (acc, (b, w)) => sigmoid(w * acc + b)
    }
  }

  def evaluate(biases: BiasSeq, weights: WeightSeq): (Int, Long) = {
    val testResults = testData.map {
      case (x, y) =>
        val output = feedForward(x, biases, weights)
        val maxOutput = argmax(output)
        val maxTarget = argmax(y)
        if (maxTarget == maxOutput) 1 else 0
    }
    val epochTime = System.nanoTime()
    (testResults.sum, epochTime - startTime)
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

  def receive = {
    case EvaluateModel =>
      if (counter != 0 && counter % parallelFactor == 0) {
        implicit val timeout = Timeout(5 seconds)
        val parameterFuture = parameterServer ? FetchParameters
        val parameterTuple = Await.result(parameterFuture, timeout.duration).asInstanceOf[ParameterTuple]
        val (biases, weights) = parameterTuple
        val resultTuple = evaluate(biases, weights)
        results = results :+ resultTuple
        counter = counter + 1
        log.info(s"EVALUATION ${resultTuple._1} / ${testData.length}")
      } else {
        counter = counter + 1
      }

    case StartTimer =>
      startTime = System.nanoTime()

    case EvaluatorDone =>
      if (doneCounter + 1 == parallelFactor) {
        log.info("Evaluator writing results")
        val outputFile = new File("/tmp/netoutput/output.csv")
        outputFile.createNewFile()
        printToFile(outputFile) { p =>
          results.foreach(tuple => p.println(tuple.productIterator.mkString(",")))
        }
        context.parent ! MasterDone
      } else {
        doneCounter = doneCounter + 1
      }

  }

}
