package downpour

import akka.actor.{Actor, ActorLogging, ActorRef}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import downpour.ParameterServer.{FetchParameters, PushGradient}
import downpour.Replica.{GetMiniBatch, GetParameters}
import downpour.Types.{ParameterTuple, TrainingExample, TrainingTupleVector}
import scala.concurrent.duration._
import scala.concurrent.Await
import akka.pattern.ask
import akka.util.Timeout
import downpour.DataShard.FetchNewBatch



object Replica {
  case class GetParameters(parameters: ParameterTuple)
  case class GetMiniBatch(miniBatch: TrainingTupleVector)
}

class Replica(parameterServer: ActorRef,
              numLayers: Int) extends Actor with ActorLogging {

  def processMiniBatch(miniBatch: TrainingTupleVector, parameters: ParameterTuple) : ParameterTuple = {
    val (biases, weights) = parameters
    var nablaB = biases.map {b => DenseVector.zeros[Double](b.length)}
    var nablaW = weights.map {w => DenseMatrix.zeros[Double](w.rows, w.cols)}
    miniBatch.foreach {
      case (x, y) =>
        val (deltaNablaB, deltaNablaW) = backpropagation(x, y, parameters)
        nablaB = nablaB.zip(deltaNablaB).map {
          case (nb, dnb) => nb + dnb
        }
        nablaW = nablaW.zip(deltaNablaW).map {
          case (nw, dnw) => nw + dnw
        }
    }
    (nablaB, nablaW)
  }

  def costDerivative(outputActivations:TrainingExample, y: TrainingExample): TrainingExample = {
    outputActivations - y
  }

  def backpropagation(x: TrainingExample, y: TrainingExample, parameters: ParameterTuple) : ParameterTuple = {
    val (biases, weights) = parameters
    var nablaB = biases.map {b => DenseVector.zeros[Double](b.length)}
    var nablaW = weights.map {w => DenseMatrix.zeros[Double](w.rows, w.cols)}
    var activation = x
    var activations: Vector[DenseVector[Double]] = Vector(x)
    var zs = Vector.empty[DenseVector[Double]]
    biases.zip(weights).foreach {
      case (b, w) =>
        val tempVar = w * activation
        val z = tempVar + b
        zs = zs :+ z
        activation = sigmoid(z)
        activations = activations :+ activation
    }
    var delta = costDerivative(activations.reverse(0), y) :* sigmoidder(zs.reverse(0))
    nablaB = nablaB.reverse.patch(0, Seq(delta), 1).reverse
    val replacementW = delta * activations.reverse(1).t
    nablaW = nablaW.reverse.patch(0, Seq(replacementW), 1).reverse
    (2 until numLayers).foreach { l =>
      val index = l - 1
      val z = zs.reverse(index)
      val sp = sigmoidder(z)
      delta = (weights.reverse(index - 1).t * delta) :* sp
      nablaB = nablaB.reverse.patch(index, Seq(delta), 1).reverse
      val anotherW = delta * activations.reverse(index + 1).t
      nablaW = nablaW.reverse.patch(index, Seq(anotherW), 1).reverse
    }
    (nablaB, nablaW)
  }

//  log.info("REQUESTING NEW BATCH")
  context.parent ! FetchNewBatch

  def receive = {

    case GetMiniBatch(batch) =>
      implicit val timeout = Timeout(5 seconds)
      val parametersFuture = parameterServer ? FetchParameters
      val parameters = Await.result(parametersFuture, timeout.duration).asInstanceOf[ParameterTuple]
      val nablaTuple = processMiniBatch(batch, parameters)
//      log.info("PUSHING GRADIENTS")
      parameterServer ! PushGradient(nablaTuple)
      context.parent ! FetchNewBatch

  }

}
