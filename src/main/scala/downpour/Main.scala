package downpour

object Main {
  def main(args: Array[String]): Unit = {
    akka.Main.main(Array(classOf[Master].getName))
  }
}
