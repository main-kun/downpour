package downpour

import breeze.generic.{MappingUFunc, UFunc}
import breeze.numerics.sigmoid

object sigmoidder extends UFunc with MappingUFunc {
  implicit object sigmoidderImplInt extends Impl[Int, Double] {
    def apply(x: Int) = sigmoid(x) * (1d - sigmoid(x))
  }

  implicit object sigmoidderImplDouble extends Impl[Double, Double] {
    def apply(x: Double) = sigmoid(x) * (1d - sigmoid(x))
  }

  implicit object sigmoidImplFloat extends Impl[Float, Float] {
    def apply(x: Float) = sigmoid(x) * (1f - sigmoid(x))
  }
}
