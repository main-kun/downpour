package downpour

import akka.actor.{Actor, ActorRef, Props}
import downpour.Master.Done
import downpour.Types.{ParameterTuple, TrainingTupleVector, TrainingVector}

object Master {
  case class Done()
}

class Master extends Actor {
  val mnist: MnistDataset = Mnist.trainDataset
  val mnistTest = Mnist.testDataset

  val trainImages: TrainingVector = mnist.imagesAsVectors.take(50000).toVector
  val trainLabels: TrainingVector = mnist.labelsAsVectors.take(50000).toVector
  val testImages: TrainingVector = mnistTest.imagesAsVectors.take(10000).toVector
  val testLabels: TrainingVector = mnistTest.labelsAsVectors.take(10000).toVector

  val zippedTrain: TrainingTupleVector = trainImages.zip(trainLabels)
  val zippedTest:TrainingTupleVector = testImages.zip(testLabels)


  val miniBatchSize = 10
  val learningRate = 3.0
  val dimensions = Seq(784, 30, 10)

  val parameterServer: ActorRef = context.actorOf(Props(new ParameterServer(
    dimensions = dimensions,
    learningRate = learningRate,
    miniBatchSize = miniBatchSize
  )))

  val evaluator: ActorRef = context.actorOf(Props(
    new Evaluator(zippedTest, parameterServer)
  ))

  val dataShard: ActorRef = context.actorOf(Props(new DataShard(
    zippedTrain,
    miniBatchSize,
    1,
    dimensions.length,
    parameterServer,
    evaluator
  )))


  def receive = {
    case Done =>
      println("done")
  }


//  parameterServer ! FetchParameters
//
//  def receive = {
//    case GetBatch(batch) =>
//      println("GOT BATCH")
//      println(batch.length)
//
//    case GetParameters(parameters) =>
//      println("GOT PARAMETERS")
//      println(parameters._1)
//      context.sender() ! PushGradient(parameters)
//  }
}
