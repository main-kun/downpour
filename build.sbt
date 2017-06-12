name := "Downpour"

version := "1.0"

scalaVersion := "2.11.6"
resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.12"
libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.5.1"
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.12"