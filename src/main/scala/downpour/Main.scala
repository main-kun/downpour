package downpour

import akka.actor.{ActorSystem, Props}

object Main extends App {
  val system = ActorSystem("downpour")
  val parallelFactor = util.Properties.envOrElse("PARALLEL_FACTOR", "2").toInt
  val numEpochs = util.Properties.envOrElse("NUM_EPOCHS", "30").toInt
  val master = system.actorOf(Props(new Master(parallelFactor)))
}
