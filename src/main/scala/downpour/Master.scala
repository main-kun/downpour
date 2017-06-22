package downpour

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import breeze.linalg.DenseVector
import downpour.Master.MasterDone
import downpour.Types.{ParameterTuple, TrainingTupleVector, TrainingVector}

import scala.collection.immutable.IndexedSeq

/**
  * Master - Actor creating the system for Downpour SGD. Master prepares MNIST data,
  * sets miniBatchSize, learningRate, network dimensions, and number of replicas
  *
  * @constructor Create Master actor and subsequently the whole actor system
  */

object Master {
  case class MasterDone()
}

class Master(parallelFactor: Int = 2, numEpochs: Int = 30) extends Actor with ActorLogging {
  val mnist: MnistDataset = Mnist.trainDataset
  val mnistTest: MnistDataset = Mnist.testDataset

  val trainImages: TrainingVector = mnist.imagesAsVectors.take(50000).toVector
  val trainLabels: TrainingVector = mnist.labelsAsVectors.take(50000).toVector
  val testImages: TrainingVector = mnistTest.imagesAsVectors.take(10000).toVector
  val testLabels: TrainingVector = mnistTest.labelsAsVectors.take(10000).toVector

  val zippedTrain: TrainingTupleVector = trainImages.zip(trainLabels)
  val zippedTest: TrainingTupleVector = testImages.zip(testLabels)


  val miniBatchSize = 10
  val learningRate = 3.0
  val dimensions = Seq(784, 30, 10)
  val numDataPerShard: Int = zippedTrain.length / parallelFactor
  val dataByShard: IndexedSeq[TrainingTupleVector] = (zippedTrain.indices by numDataPerShard).map {
    k => zippedTrain.slice(k, k + numDataPerShard)
  }

  val parameterServer: ActorRef = context.actorOf(Props(new ParameterServer(
    dimensions = dimensions,
    learningRate = learningRate,
    miniBatchSize = miniBatchSize
  )))

  val evaluator: ActorRef = context.actorOf(Props(
    new Evaluator(zippedTest, parameterServer, parallelFactor)
  ))

  log.info(s"Starting $parallelFactor replicas")
  log.info(s"$numEpochs epochs")
  var dataShards: IndexedSeq[ActorRef] = (0 until parallelFactor).map { i =>
    context.actorOf(Props(new DataShard(
      dataByShard(i),
      miniBatchSize,
      i,
      dimensions.length,
      numEpochs,
      parameterServer,
      evaluator
    )))
  }


  def receive = {
    case MasterDone =>
      log.info("Master done")
      context.system.terminate()
  }


}
