package com.cap

import com.cap.TextSentAnalytics.{getNgram, standardiseString}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.streaming.twitter._
import org.apache.spark.streaming.{Seconds, StreamingContext};

/**
 * Created by Lewis Gavin 06/09/2016
 *
 */
case object SarcasmDetector {

  def main (args: Array[String]) {


    val consumerKey = args(0)
    val consumerSecret = args(1)
    val accessToken = args(2)
    val accessTokenSecret = args(3)

    val sc = new SparkContext()
    val sqlContext = new SQLContext(sc)


    val ssc = new StreamingContext(sc, Seconds(30))

    System.setProperty("twitter4j.oauth.consumerKey", consumerKey)
    System.setProperty("twitter4j.oauth.consumerSecret", consumerSecret)
    System.setProperty("twitter4j.oauth.accessToken", accessToken)
    System.setProperty("twitter4j.oauth.accessTokenSecret", accessTokenSecret)

    val stream = TwitterUtils.createStream(ssc, None, Seq("sarcastic"))

    //extract sarcastic tweets and standardise the string
    val sarcasmTweets = stream.filter {tweets =>
      val tags = tweets.getText.split(" ").filter(_.startsWith("#")).map(_.toLowerCase)
      tags.contains("#sarcastic")
    }.map(tweet => standardiseString(tweet.getText.replace("sarcastic","")))
      .map(tweet => (1, getNgram(tweet, 2)))

    //extract non sarcastic tweets and standardise the string
    val tweets = stream.filter {t =>
      val tags = t.getText.split(" ").filter(_.startsWith("#")).map(_.toLowerCase)
      !tags.contains("#sarcastic")
    }.map(tweet => standardiseString(tweet.getText))
      .map(tweet => (0, getNgram(tweet, 2)))

    sarcasmTweets.foreachRDD(rdd => rdd.take(5).foreach(println))

    ssc.start()
    ssc.awaitTermination()

    /*

    //read the file into a data frame
    val trainingDF = sqlContext.createDataFrame(sc.textFile(args(0)).map(input => input.split(",")).map(x => (x(0),x(1).split(" ")))).toDF("label","text")
    val nonTrainedDF = sqlContext.createDataFrame(sc.textFile(args(1)).map(input => input.split(",")).map(x => (x(0),x(1).split(" ")))).toDF("label","text")


    // convert words to a vector to be used within the Naive Bayes
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(4)
      .setMinCount(0)
    val training = word2Vec.fit(trainingDF).transform(trainingDF)
    val nonTraining = word2Vec.fit(nonTrainedDF).transform(nonTrainedDF)


    // create a labelled point from the label and vector (each number in the vector needs to be converted to its absolute value as Naive Bayes doesnt accept negatives)l
    val trainingData = training.map { line =>
      LabeledPoint(line.getAs[String]("label").toDouble, Vectors.dense(line.getAs[org.apache.spark.mllib.linalg.Vector]("result").toDense.toArray.map(a => Math.abs(a))))
    }
    val nonTrainData = nonTraining.map { line =>
      LabeledPoint(line.getAs[String]("label").toDouble, Vectors.dense(line.getAs[org.apache.spark.mllib.linalg.Vector]("result").toDense.toArray.map(a => Math.abs(a))))
    }

    // Split data into training (60%) and test (40%).
    //val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)

    val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = nonTrainData.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / nonTrainData.count()

    predictionAndLabel.collect().foreach(println)
    println("Accuracy " + accuracy)

    model.save(sc, "file:///home/cloudera/Documents/SarcasmModel")
    // Save and load model
   // model.save(sc, "target/tmp/myNaiveBayesModel")
    //val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")

    */

  }
}
