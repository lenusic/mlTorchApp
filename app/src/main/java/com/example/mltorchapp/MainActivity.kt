package com.example.mltorchapp

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.*

class MainActivity : AppCompatActivity(), Runnable {

    private var mModule: Module? = null

    private lateinit var mResultTextView: TextView
    private lateinit var mRecognizeButton: Button
    private lateinit var mClearButton: Button
    private lateinit var mDrawView: DrawingView

    private val indexToChar = mapOf(
        0 to ' ', 1 to 'a', 2 to 'b', 3 to 'c', 4 to 'd', 5 to 'e', 6 to 'f', 7 to 'g', 8 to 'h',
        9 to 'i', 10 to 'j', 11 to 'k', 12 to 'l', 13 to 'm', 14 to 'n', 15 to 'o', 16 to 'p', 17 to 'q',
        18 to 'r', 19 to 's', 20 to 't', 21 to 'u', 22 to 'v', 23 to 'w', 24 to 'x', 25 to 'y', 26 to 'z',
        27 to '0', 28 to '1', 29 to '2', 30 to '3', 31 to '4', 32 to '5', 33 to '6', 34 to '7', 35 to '8',
        36 to '9', 37 to ',', 38 to '.', 39 to '?', 40 to '!'
    )


    companion object {
        private const val EXPECTED_SIZE = 25  // Number of strokes/Bezier curves
        private const val FEATURE_SIZE = 29  // Number of features per stroke
        private const val EXPECTED_INPUT_SIZE = EXPECTED_SIZE * FEATURE_SIZE  // Total input size

        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }

            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            return file.absolutePath
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mResultTextView = findViewById(R.id.resultTextView)
        mDrawView = findViewById(R.id.drawview)
        mRecognizeButton = findViewById(R.id.recognizeButton)
        mClearButton = findViewById(R.id.clearButton)

        mRecognizeButton.setOnClickListener {
            val thread = Thread(this@MainActivity)
            thread.start()
        }

        mClearButton.setOnClickListener {
            mResultTextView.text = ""
            mDrawView.clearCanvas()
        }

        try {
            mModule = LiteModuleLoader.load(assetFilePath(this, "torchscript_model.ptl"))
        } catch (e: Exception) {
            Log.e("PytorchDemo", "Error reading assets", e)
            finish()
        }
    }

    override fun run() {
        val result = processDrawingAndPredict()
        result?.let { prediction ->
            runOnUiThread {
                mResultTextView.text = prediction
                mDrawView.clearCanvas()
            }
        }
    }


    private fun processDrawingAndPredict(): String? {
        val strokes = mDrawView.getAllPoints()
        if (strokes.isEmpty()) return null

        var flattenedPoints = mutableListOf<Float>()

        try {
            for (stroke in strokes) {
                val xPoints = stroke.map { it.first }
                val yPoints = stroke.map { it.second }
                val timeStamps = stroke.map { it.third.toFloat() }

                // Normalize coordinates to fit within [0, 1]
                val minX = xPoints.minOrNull() ?: 0f
                val maxX = xPoints.maxOrNull() ?: 1f
                val minY = yPoints.minOrNull() ?: 0f
                val maxY = yPoints.maxOrNull() ?: 1f

                val normalizedX = xPoints.map { (it - minX) / (maxX - minX).coerceAtLeast(1e-9f) }
                val normalizedY = yPoints.map { (it - minY) / (maxY - minY).coerceAtLeast(1e-9f) }

                // Calculate features
                val totalLength = calculateTotalLength(normalizedX, normalizedY)
                val directness = calculateDirectness(normalizedX, normalizedY)
                val totalCurvature = calculateTotalCurvature(normalizedX, normalizedY)
                val (avgSinDirection, avgCosDirection) = calculateAverageSinCosDirection(normalizedX, normalizedY)
                val (avgSinCurvature, avgCosCurvature) = calculateAverageSinCosCurvature(normalizedX, normalizedY)
                val endPointDiff = calculateEndPointDiff(normalizedX, normalizedY)
                val controlPointDistributions = calculateControlPointDistributions(normalizedX, normalizedY, endPointDiff)
                val angles = calculateAngles(normalizedX, normalizedY)
                val timeCoefficients = calculateTimeCoefficients(timeStamps)
                val penUpFlag = 1f  // Adjust if you have pen-up information

                // Flatten points for model input
                flattenedPoints.addAll(listOf(totalLength, directness, totalCurvature, avgSinDirection, avgCosDirection, avgSinCurvature, avgCosCurvature))
                flattenedPoints.addAll(endPointDiff)
                flattenedPoints.addAll(controlPointDistributions)
                flattenedPoints.addAll(angles)
                flattenedPoints.addAll(timeCoefficients)
                flattenedPoints.add(penUpFlag)
            }

            // Ensure the serialized data matches the expected size
            if (flattenedPoints.size < EXPECTED_INPUT_SIZE) {
                flattenedPoints.addAll(List(EXPECTED_INPUT_SIZE - flattenedPoints.size) { 0f })
            } else if (flattenedPoints.size > EXPECTED_INPUT_SIZE) {
                flattenedPoints = flattenedPoints.take(EXPECTED_INPUT_SIZE).toMutableList()
            }

            return predictWithTensor(flattenedPoints.toFloatArray())
        } catch (e: Exception) {
            Log.e("PytorchDemo", "Error during preprocessing or inference", e)
            return null
        }
    }




//    private fun predictWithTensor(flattenedPoints: FloatArray): String? {
//        // Check for NaN values in the flattenedPoints array and replace them with 0f
//        for (i in flattenedPoints.indices) {
//            if (flattenedPoints[i].isNaN()) {
//                flattenedPoints[i] = 0f
//            }
//        }
//
//        // Check for extreme values and log them
//        val maxVal = flattenedPoints.maxOrNull()
//        val minVal = flattenedPoints.minOrNull()
//        Log.d("PytorchDemo", "Max value in input tensor: $maxVal")
//        Log.d("PytorchDemo", "Min value in input tensor: $minVal")
//
//        Log.d("PytorchDemo", "Input Tensor: ${flattenedPoints.joinToString(", ")}")
//        val inputTensor = Tensor.fromBlob(flattenedPoints, longArrayOf(1, EXPECTED_SIZE.toLong(), FEATURE_SIZE.toLong()))
//        try {
//            val modelOutput = mModule?.forward(IValue.from(inputTensor))?.toTensor()
//            val outputIndices = modelOutput?.dataAsFloatArray ?: return null
//            Log.d("PytorchDemo", "Model Output Indices: ${outputIndices.joinToString(", ")}")
//            return processModelOutput(outputIndices)
//        } catch (e: Exception) {
//            Log.e("PytorchDemo", "Error during model inference", e)
//            return null
//        }
//    }

    private fun predictWithTensor(flattenedPoints: FloatArray): String? {
        val inputTensor = Tensor.fromBlob(flattenedPoints, longArrayOf(1, EXPECTED_SIZE.toLong(), FEATURE_SIZE.toLong()))
        return try {
            val modelOutput = mModule?.forward(IValue.from(inputTensor))?.toTensor()
            val outputIndices = modelOutput?.dataAsFloatArray ?: return null
            processModelOutput(outputIndices)
        } catch (e: Exception) {
            Log.e("PytorchDemo", "Error during model inference", e)
            null
        }
    }


    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expScores = logits.map { exp(it - maxLogit) }
        val sumExpScores = expScores.sum()
        return expScores.map { it / sumExpScores }.toFloatArray()
    }

    private fun processModelOutput(outputs: FloatArray): String {
        val stringBuilder = StringBuilder()
        val numClasses = 41  // Number of output classes

        for (i in outputs.indices step numClasses) {
            val classLogits = outputs.sliceArray(i until i + numClasses)
            val classProbabilities = softmax(classLogits)
            val maxIndex = classProbabilities.indices.maxByOrNull { classProbabilities[it] } ?: 0
            val character = indexToChar[maxIndex] ?: '?'
            stringBuilder.append(character)
        }

        return stringBuilder.toString()
    }


    private fun calculateTotalLength(xPoints: List<Float>, yPoints: List<Float>): Float {
        var length = 0f
        for (i in 1 until xPoints.size) {
            val dx = xPoints[i] - xPoints[i - 1]
            val dy = yPoints[i] - yPoints[i - 1]
            length += sqrt(dx * dx + dy * dy)
        }
        return length
    }

    private fun calculateDirectness(xPoints: List<Float>, yPoints: List<Float>): Float {
        val dx = xPoints.last() - xPoints.first()
        val dy = yPoints.last() - yPoints.first()
        val directDistance = sqrt(dx * dx + dy * dy)
        val totalLength = calculateTotalLength(xPoints, yPoints)
        return if (totalLength != 0f) directDistance / totalLength else 0f
    }

    private fun calculateEndPointDiff(xPoints: List<Float>, yPoints: List<Float>): List<Float> {
        val dx = xPoints.last() - xPoints.first()
        val dy = yPoints.last() - yPoints.first()
        return listOf(dx, dy)
    }

    private fun calculateControlPointDistributions(xPoints: List<Float>, yPoints: List<Float>, endPointDiff: List<Float>): List<Float> {
        val (dx, dy) = endPointDiff
        val endPointDiffNorm = max(sqrt(dx * dx + dy * dy), 0.001f)
        return (1 until xPoints.size - 1).map {
            val controlDx = xPoints[it] - xPoints.first()
            val controlDy = yPoints[it] - yPoints.first()
            sqrt(controlDx * controlDx + controlDy * controlDy) / endPointDiffNorm
        }
    }

    private fun calculateAngles(xPoints: List<Float>, yPoints: List<Float>): List<Float> {
        return (1 until xPoints.size - 1).map {
            atan2(yPoints[it] - yPoints.first(), xPoints[it] - xPoints.first())
        }
    }

    private fun calculateTimeCoefficients(timeStamps: List<Float>): List<Float> {
        val timeRange = timeStamps.last() - timeStamps.first()
        return (1 until timeStamps.size - 1).map {
            (timeStamps[it] - timeStamps.first()) / timeRange
        }
    }

    private fun calculateTotalCurvature(xPoints: List<Float>, yPoints: List<Float>): Float {
        var totalCurvature = 0f
        for (i in 2 until xPoints.size) {
            val dx1 = xPoints[i - 1] - xPoints[i - 2]
            val dy1 = yPoints[i - 1] - yPoints[i - 2]
            val dx2 = xPoints[i] - xPoints[i - 1]
            val dy2 = yPoints[i] - yPoints[i - 1]
            val angle1 = atan2(dy1, dx1)
            val angle2 = atan2(dy2, dx2)
            totalCurvature += abs(angle2 - angle1)
        }
        return totalCurvature
    }

    private fun calculateAverageSinCosDirection(xPoints: List<Float>, yPoints: List<Float>): Pair<Float, Float> {
        val directions = (1 until xPoints.size).map {
            val dx = xPoints[it] - xPoints[it - 1]
            val dy = yPoints[it] - yPoints[it - 1]
            atan2(dy, dx)
        }
        val sinSum = directions.sumOf { sin(it).toDouble() }
        val cosSum = directions.sumOf { cos(it).toDouble() }

        return Pair((sinSum / directions.size).toFloat(), (cosSum / directions.size).toFloat())
    }

    private fun calculateAverageSinCosCurvature(xPoints: List<Float>, yPoints: List<Float>): Pair<Float, Float> {
        val curvatures = (2 until xPoints.size).map {
            val dx1 = xPoints[it - 1] - xPoints[it - 2]
            val dy1 = yPoints[it - 1] - yPoints[it - 2]
            val dx2 = xPoints[it] - xPoints[it - 1]
            val dy2 = yPoints[it] - yPoints[it - 1]
            val angle1 = atan2(dy1, dx1)
            val angle2 = atan2(dy2, dx2)
            abs(angle2 - angle1)
        }
        val sinSum = curvatures.sumOf { sin(it).toDouble() }
        val cosSum = curvatures.sumOf { cos(it).toDouble() }

        return Pair((sinSum / curvatures.size).toFloat(), (cosSum / curvatures.size).toFloat())
    }


}
