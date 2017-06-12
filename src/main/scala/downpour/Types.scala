package downpour

import breeze.linalg.{DenseMatrix, DenseVector}

object Types {
  type ParameterTuple = (Seq[DenseVector[Double]], Seq[DenseMatrix[Double]])
  type TrainingExample = DenseVector[Double]
  type BiasSeq = Seq[DenseVector[Double]]
  type WeightSeq = Seq[DenseMatrix[Double]]
  type TrainingVector = Vector[DenseVector[Double]]
  type TrainingTupleVector = Vector[(DenseVector[Double], DenseVector[Double])]
}
