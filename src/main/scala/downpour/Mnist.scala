package downpour
/*
  MIT License

  Copyright (c) 2017 Alexey Noskov

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
  persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
  Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
import java.io.{ File, FileInputStream, FileOutputStream, DataInputStream }
import java.net.URL
import java.nio.file.{ Files, Paths }
import java.nio.channels.Channels
import java.util.zip.GZIPInputStream
import breeze.linalg._

class MnistFileReader(location: String, fileName: String) {

  private[this] val path = Paths.get(location, fileName)

  if (!Files.exists(path))
    download()

  protected[this] val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path.toString)))

  private def download(): Unit = {
    val rbc = Channels.newChannel(new URL(s"http://yann.lecun.com/exdb/mnist/$fileName").openStream())
    val fos = new FileOutputStream(s"$location/$fileName")
    fos.getChannel.transferFrom(rbc, 0, Long.MaxValue)
  }

}

class MnistLabelReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count: Int = stream.readInt()

  val labelsAsInts: Stream[Int] = readLabels(0)
  val labelsAsVectors: Stream[DenseVector[Double]] = labelsAsInts.map { label =>
    DenseVector.tabulate[Double](10) { i => if (i == label) 1.0 else 0.0 }
  }

  private[this] def readLabels(ind: Int): Stream[Int] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(stream.readByte(), readLabels(ind + 1))

}

class MnistImageReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count: Int = stream.readInt()
  val width: Int = stream.readInt()
  val height: Int = stream.readInt()

  val imagesAsMatrices: Stream[DenseMatrix[Int]] = readImages(0)
  val imagesAsVectors: Stream[DenseVector[Double]] = imagesAsMatrices map { image =>
    DenseVector.tabulate(width * height) { i => image(i / width, i % height) / 255.0 }
  }

  private[this] def readImages(ind: Int): Stream[DenseMatrix[Int]] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(readImage(), readImages(ind + 1))

  private[this] def readImage(): DenseMatrix[Int] = {
    val m = DenseMatrix.zeros[Int](height, width)

    for (y <- 0 until height; x <- 0 until width)
      m(y, x) = stream.readUnsignedByte()

    m
  }

}

class MnistDataset(location: String, dataset: String) {

  lazy val imageReader = new MnistImageReader(location, s"$dataset-images-idx3-ubyte.gz")
  lazy val labelReader = new MnistLabelReader(location, s"$dataset-labels-idx1-ubyte.gz")

  def imageWidth: Int = imageReader.width
  def imageHeight:Int = imageReader.height

  def imagesAsMatrices:Stream[DenseMatrix[Int]]= imageReader.imagesAsMatrices
  def imagesAsVectors:Stream[DenseVector[Double]] = imageReader.imagesAsVectors

  def labelsAsInts: Stream[Int] = labelReader.labelsAsInts
  def labelsAsVectors: Stream[DenseVector[Double]] = labelReader.labelsAsVectors

  def examples: Stream[(DenseVector[Double], DenseVector[Double])] = imagesAsVectors zip labelsAsVectors

}

object Mnist {

  val location: String = Option(System.getenv("MNIST_PATH")).getOrElse(List(System.getenv("HOME"), ".cache", "mnist").mkString(File.separator))
  val locationFile = new File(location)

  if (!locationFile.exists)
    locationFile.mkdirs

  val trainDataset = new MnistDataset(location, "train")
  val testDataset = new MnistDataset(location, "t10k")

}
