package com.cap

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext

/**
 * Created by Lewis Gavin 06/09/2016
 *
 */
case object SarcasmDetector {

  def main (args: Array[String]) {

    val sc = new SparkContext()
    val sqlContext = new SQLContext(sc)
    val documentDF = sqlContext.createDataFrame(sc.textFile(args(0)).map(input => input.split(",")).map(x => (x(0),x(1).split(" ")))).toDF("label","text")

    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val result = word2Vec.fit(documentDF).transform(documentDF)


    result.printSchema()

    val parsedData = result.map { line =>
      LabeledPoint(line.getAs[String]("label").toDouble, Vectors.dense(line.getAs[org.apache.spark.mllib.linalg.Vector]("result").toDense.toArray.map(a => Math.abs(a))))
    }

    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    predictionAndLabel.collect().foreach(println)
    println("Accuracy " + accuracy)

    model.save(sc, "file:///home/cloudera/Documents/SarcasmModel")
    // Save and load model
   // model.save(sc, "target/tmp/myNaiveBayesModel")
    //val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")

  }
}
