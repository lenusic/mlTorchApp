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

    private lateinit var rawData: FloatArray

    private var mModule: Module? = null

    private lateinit var mResultTextView: TextView
    private lateinit var mRecognizeButton: Button
    private lateinit var mClearButton: Button
    private lateinit var mDrawView: DrawingView

    private val indexToChar = mapOf(
        0 to ' ', 1 to '!', 2 to '"', 3 to '#', 4 to '&', 5 to '\'', 6 to '(', 7 to ')', 8 to '*',
        9 to '+', 10 to ',', 11 to '-', 12 to '.', 13 to '/', 14 to '0', 15 to '1', 16 to '2',
        17 to '3', 18 to '4', 19 to '5', 20 to '6', 21 to '7', 22 to '8', 23 to '9', 24 to ':',
        25 to ';', 26 to '?', 27 to 'A', 28 to 'B', 29 to 'C', 30 to 'D', 31 to 'E', 32 to 'F',
        33 to 'G', 34 to 'H', 35 to 'I', 36 to 'J', 37 to 'K', 38 to 'L', 39 to 'M', 40 to 'N',
        41 to 'O', 42 to 'P', 43 to 'Q', 44 to 'R', 45 to 'S', 46 to 'T', 47 to 'U', 48 to 'V',
        49 to 'W', 50 to 'X', 51 to 'Y', 52 to 'Z', 53 to '[', 54 to ']', 55 to 'a', 56 to 'b',
        57 to 'c', 58 to 'd', 59 to 'e', 60 to 'f', 61 to 'g', 62 to 'h', 63 to 'i', 64 to 'j',
        65 to 'k', 66 to 'l', 67 to 'm', 68 to 'n', 69 to 'o', 70 to 'p', 71 to 'q', 72 to 'r',
        73 to 's', 74 to 't', 75 to 'u', 76 to 'v', 77 to 'w', 78 to 'x', 79 to 'y', 80 to 'z'
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
                mResultTextView.append(prediction)  // Append the character to the result view
                mDrawView.clearCanvas()  // Clear canvas if needed after each character
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

            // Clean the data to avoid NaNs and Infinities
            val cleanedData = cleanData(flattenedPoints.toFloatArray())

            return predictWithTensor(cleanedData)
        } catch (e: Exception) {
            Log.e("PytorchDemo", "Error during preprocessing or inference", e)
            return null
        }
    }

    private fun cleanData(inputData: FloatArray): FloatArray {
        return inputData.map {
            when {
                it.isNaN() -> 0.0f
                it.isInfinite() -> 0.0f
                else -> it
            }
        }.toFloatArray()
    }



    private fun predictWithTensor(flattenedPoints: FloatArray): String? {
        val inputTensor = Tensor.fromBlob(flattenedPoints, longArrayOf(1, EXPECTED_SIZE.toLong(), FEATURE_SIZE.toLong()))
        Log.d("PytorchDemo", "Input Tensor Shape: ${inputTensor.shape().joinToString()}")
        Log.d("PytorchDemo", "Input Tensor Values: ${flattenedPoints.joinToString()}")
        return try {
            val modelOutput = mModule?.forward(IValue.from(inputTensor))?.toTensor()
            Log.d("PytorchDemo", "Model Output Shape: ${modelOutput?.shape()?.joinToString()}")
            val outputIndices = modelOutput?.dataAsFloatArray ?: return null
            Log.d("PytorchDemo", "Logits before softmax: ${outputIndices.joinToString()}")

            // Ensure that the output size is correct
            val expectedOutputSize = 81  // Adjust to the number of classes in your model
            if (outputIndices.size != expectedOutputSize * EXPECTED_SIZE) {
                Log.e("PytorchDemo", "Unexpected output size: ${outputIndices.size}, expected: ${expectedOutputSize * EXPECTED_SIZE}")
                return "?"  // Indicate an error with a placeholder character
            }

            val decodedResult = processModelOutput(outputIndices)
            Log.d("PytorchDemo", "Decoded result: $decodedResult")
            decodedResult
        } catch (e: Exception) {
            Log.e("PytorchDemo", "Error during model inference", e)
            null
        }
    }


    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: Float.NEGATIVE_INFINITY
        val exps = logits.map { Math.exp((it - maxLogit).toDouble()).toFloat() }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    private fun processModelOutput(outputs: FloatArray): String {
        val numClasses = 81  // Number of output classes
        val classProbabilities = softmax(outputs)
        val maxIndex = classProbabilities.indices.maxByOrNull { classProbabilities[it] } ?: 0
        val character = indexToChar[maxIndex] ?: '?'

        // Ensure maxIndex is within the expected range
        if (maxIndex < 0 || maxIndex >= numClasses) {
            Log.e("ModelOutput", "Invalid index: $maxIndex")
            return "?"
        }

        // Check if maxIndex is within indexToChar keys
        if (!indexToChar.containsKey(maxIndex)) {
            Log.e("ModelOutput", "maxIndex not found in indexToChar map: $maxIndex")
            return "?"
        }

        // Log the outputs for debugging
        Log.d("ModelOutput", "Logits: ${outputs.joinToString(", ")}")
        Log.d("ModelOutput", "Softmax: ${classProbabilities.joinToString(", ")}")
        Log.d("ModelOutput", "Character: $character")
        Log.d("ModelOutput", "MaxIndex: $maxIndex, Character: $character")

        // Return the single most likely character
        return character.toString()
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
