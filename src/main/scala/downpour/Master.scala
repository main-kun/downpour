package downpour

import akka.actor.{Actor, ActorRef, Props}
import downpour.Master.Done
import downpour.Types.{ParameterTuple, TrainingTupleVector, TrainingVector}

object Master {
  case class Done()
}

class Master extends Actor {
  val mnist: MnistDataset = Mnist.trainDataset
  val trainImages: TrainingVector = mnist.imagesAsVectors.take(100).toVector
  val trainLabels: TrainingVector = mnist.labelsAsVectors.take(100).toVector
  val zippedTrain: TrainingTupleVector = trainImages.zip(trainLabels)

  val miniBatchSize = 10
  val learningRate = 3.0
  val dimensions = Seq(784, 30, 10)

  val parameterServer: ActorRef = context.actorOf(Props(new ParameterServer(
    dimensions = dimensions,
    learningRate = learningRate,
    miniBatchSize = miniBatchSize
  )))

  val dataShard: ActorRef = context.actorOf(Props(new DataShard(
    zippedTrain,
    miniBatchSize,
    1,
    dimensions.length,
    parameterServer
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
