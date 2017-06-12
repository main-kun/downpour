package downpour

import akka.actor.{Actor, ActorLogging}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gaussian
import downpour.ParameterServer.{FetchParameters, PushGradient}
import downpour.Types.{BiasSeq, ParameterTuple, WeightSeq}

object ParameterServer {
  case class FetchParameters()
  case class PushGradient(nablaTuple: ParameterTuple)
}

class ParameterServer(dimensions: Seq[Int],
                      learningRate: Double,
                      miniBatchSize: Int) extends Actor with ActorLogging {
  var normalDist = Gaussian(0,1)

  var biases: BiasSeq = dimensions.tail.map { x => DenseVector.rand[Double](x,normalDist)}
  var weights: WeightSeq = (dimensions.dropRight(1), dimensions.drop(1)).zipped map {
    case(x,y) => DenseMatrix.rand[Double](y, x, normalDist)
  }

  def receive = {
    case FetchParameters =>
      log.info("SENDING PARAMETERS")
      context.sender() ! (biases, weights)


    case PushGradient(nablaTuple: ParameterTuple) =>
      log.info("GOT NABLAS BABY")
      val (nablaB, nablaW) = nablaTuple
      weights = weights.zip(nablaW).map {
        case (w, nw) =>
          val rightPart = nw * (learningRate/miniBatchSize)
          w - rightPart
      }
      biases = biases.zip(nablaB).map {
        case (b, nb) =>
          val rightPart = nb * (learningRate/miniBatchSize)
          b - rightPart
      }
  }

}
