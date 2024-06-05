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
        result?.let {
            runOnUiThread {
                mResultTextView.text = it
                mDrawView.clearCanvas()
            }
        }
    }

    private fun processDrawingAndPredict(): String? {
        val points = mDrawView.getAllPoints()
        if (points.isEmpty()) return null

        var flattenedPoints = mutableListOf<Float>()

        for (stroke in points) {
            if (stroke.isEmpty()) continue

            val xPoints = stroke.map { it.first }
            val yPoints = stroke.map { it.second }
            val timeStamps = stroke.map { it.third.toFloat() }

            // Normalize x and y coordinates
            val minX = xPoints.minOrNull() ?: 0f
            val maxX = xPoints.maxOrNull() ?: 1f
            val minY = yPoints.minOrNull() ?: 0f
            val maxY = yPoints.maxOrNull() ?: 1f
            val normalizedX = xPoints.map { (it - minX) / (maxX - minX).coerceAtLeast(1e-9f) }
            val normalizedY = yPoints.map { (it - minY) / (maxY - minY).coerceAtLeast(1e-9f) }

            // Calculate features
            val totalLength = calculateTotalLength(normalizedX, normalizedY)
            val directness = calculateDirectness(normalizedX, normalizedY)
            val (avgSinDir, avgCosDir) = calculateAverageSinCosDirection(normalizedX, normalizedY)
            val (avgSinCurv, avgCosCurv) = calculateAverageSinCosCurvature(normalizedX, normalizedY)
            val endPointDiff = calculateEndPointDiff(normalizedX, normalizedY)
            val controlPointDistributions = calculateControlPointDistributions(normalizedX, normalizedY, endPointDiff)
            val angles = calculateAngles(normalizedX, normalizedY)
            val timeCoefficients = calculateTimeCoefficients(timeStamps)
            val penUpFlag = if (stroke.last().third == 1L) 1f else 0f

            // Add features to list
            flattenedPoints.add(totalLength)
            flattenedPoints.add(directness)
            flattenedPoints.add(avgSinDir)
            flattenedPoints.add(avgCosDir)
            flattenedPoints.add(avgSinCurv)
            flattenedPoints.add(avgCosCurv)
            flattenedPoints.addAll(endPointDiff)
            flattenedPoints.addAll(controlPointDistributions)
            flattenedPoints.addAll(angles)
            flattenedPoints.addAll(timeCoefficients)
            flattenedPoints.add(penUpFlag)
        }

        // Ensure the serialized data matches the expected size
        val expectedSize = EXPECTED_INPUT_SIZE

        if (flattenedPoints.size < expectedSize) {
            // Pad with zeros
            flattenedPoints.addAll(List(expectedSize - flattenedPoints.size) { 0f })
            Log.d("PytorchDemo", "Serialized Stroke Data Size: ${flattenedPoints.size} (padded)")
            Log.d("PytorchDemo", "Expected Size: $expectedSize")
        } else if (flattenedPoints.size > expectedSize) {
            // Truncate the array
            flattenedPoints = flattenedPoints.take(expectedSize).toMutableList()
            Log.d("PytorchDemo", "Serialized Stroke Data Size: ${flattenedPoints.size} (truncated)")
            Log.d("PytorchDemo", "Expected Size: $expectedSize")
        }

        return predictWithTensor(flattenedPoints.toFloatArray())
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


    private fun predictWithTensor(flattenedPoints: FloatArray): String? {
        val inputTensor = Tensor.fromBlob(flattenedPoints, longArrayOf(1, EXPECTED_SIZE.toLong(), FEATURE_SIZE.toLong()))
        val modelOutput = mModule?.forward(IValue.from(inputTensor))?.toTensor()
        val outputIndices = modelOutput?.dataAsFloatArray ?: return null

        val probabilities = softmax(outputIndices)
        return processModelOutput(probabilities)
    }
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expScores = logits.map { exp(it - maxLogit) }
        val sumExpScores = expScores.sum()
        return expScores.map { it / sumExpScores }.toFloatArray()
    }

    private fun calculateTotalLength(stroke: List<Triple<Float, Float, Float>>): Float {
        return stroke.zipWithNext { a, b ->
            val dx = b.first - a.first
            val dy = b.second - a.second
            sqrt(dx * dx + dy * dy)
        }.sum()
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

    private fun decodeOutput(outputIndices: FloatArray): String {
        val stringBuilder = StringBuilder()
        val numClasses = 41  // Number of output classes

        for (i in outputIndices.indices step numClasses) {
            var maxIndex = 0
            var maxValue = outputIndices[i]
            for (j in 1 until numClasses) {
                if (outputIndices[i + j] > maxValue) {
                    maxIndex = j
                    maxValue = outputIndices[i + j]
                }
            }
            val character = indexToChar[maxIndex] ?: '?'
            stringBuilder.append(character)
        }

        return stringBuilder.toString()
    }

    private fun processModelOutput(outputs: FloatArray): String {
        val stringBuilder = StringBuilder()
        val numClasses = 41  // Number of output classes

        for (i in outputs.indices step numClasses) {
            // Find the index with the highest value for each set of numClasses values
            var maxIndex = 0
            var maxValue = outputs[i]
            for (j in 1 until numClasses) {
                if (outputs[i + j] > maxValue) {
                    maxIndex = j
                    maxValue = outputs[i + j]
                }
            }
            val character = indexToChar[maxIndex] ?: '?'
            stringBuilder.append(character)
        }

        return stringBuilder.toString()
    }

}
