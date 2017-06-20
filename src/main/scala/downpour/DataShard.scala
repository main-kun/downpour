package downpour

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import downpour.DataShard.FetchNewBatch
import downpour.Evaluator.{EvaluatorDone, EvaluateModel, StartTimer}
import downpour.Replica.GetMiniBatch
import downpour.Types.TrainingTupleVector

import scala.collection.immutable.IndexedSeq
import scala.util.Random

/**
  * DataShard - Actor that contains  training data and produces mini batches. Creates Replica actors
  *
  * @constructor Create a DataShard Actor
  *
  * @param trainingData Training dataset
  * @param miniBatchSize Size of the mini-batch
  * @param replicaId Unique ID for this replica
  * @param numLayers Number of layers in replica
  * @param numEpoch Number of epochs to compute
  * @param parameterServer ActorRef of ParameterServer
  * @param evaluator ActorRef of Evaluator
 */

object DataShard {
  case object FetchNewBatch
}

class DataShard(trainingData: TrainingTupleVector,
                miniBatchSize: Int,
                replicaId: Int,
                numLayers: Int,
                numEpoch: Int,
                parameterServer: ActorRef,
                evaluator: ActorRef) extends Actor with ActorLogging {

  var miniBatches: IndexedSeq[TrainingTupleVector] = IndexedSeq.empty
  var batchesIterator: Iterator[TrainingTupleVector] = miniBatches.iterator
  var epochCounter: Int = 0

  def generateMiniBatches(): Unit = {
    val n: Int = trainingData.length
    val shuffled: TrainingTupleVector = Random.shuffle(trainingData)
    miniBatches = (0 until n by miniBatchSize).map {
      k => shuffled.slice(k, k + miniBatchSize)
    }
    batchesIterator = miniBatches.iterator
  }

  generateMiniBatches()
  evaluator ! StartTimer
  log.info(s"Initiated $replicaId datashard")
  val replica: ActorRef = context.actorOf(Props(new Replica(parameterServer, numLayers)))

  def receive = {
    case FetchNewBatch =>
      if (batchesIterator.hasNext) {
        replica ! GetMiniBatch(batchesIterator.next())

        if(!batchesIterator.hasNext) {
          if (epochCounter == numEpoch) {
            log.info(s"Datashard $replicaId done")
            context.stop(replica)
            evaluator ! EvaluateModel
            evaluator ! EvaluatorDone
          } else {
            epochCounter += 1
            evaluator ! EvaluateModel
            generateMiniBatches()
          }
        }
      }
      else {
        generateMiniBatches()
        context.sender() ! GetMiniBatch(batchesIterator.next())
      }
  }
}
