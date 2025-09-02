# Use official OpenJDK base image
FROM openjdk:17-jdk-slim

# Set working directory
WORKDIR /app

# Copy Maven wrapper and pom.xml to build dependencies first
COPY mvnw .
COPY .mvn .mvn
COPY pom.xml .

# Download dependencies
RUN ./mvnw dependency:go-offline -B

# Copy source code
COPY src ./src

# Package the application
RUN ./mvnw clean package -DskipTests

# Run the JAR
CMD ["java", "-jar", "fraud-detection-microservice/target/fraud-detection-api-0.0.1-SNAPSHOT.jar"]

# Expose the port
EXPOSE 8080
