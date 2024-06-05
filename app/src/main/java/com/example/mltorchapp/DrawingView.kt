package com.example.mltorchapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.os.SystemClock
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class DrawingView(context: Context, attrs: AttributeSet) : View(context, attrs) {

    private var path = Path()
    private var paint = Paint()
    private var strokes = mutableListOf<Path>()
    private var currentStroke = Path()
    private var lastX = 0f
    private var lastY = 0f
    private val paths = mutableListOf<PathData>()

    init {
        paint.apply {
            color = 0xFF000000.toInt()
            style = Paint.Style.STROKE
            strokeWidth = 5f * resources.displayMetrics.density // Adjust stroke width for screen density
            isAntiAlias = true
            isDither = true
            strokeJoin = Paint.Join.ROUND
            strokeCap = Paint.Cap.ROUND
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (stroke in strokes) {
            canvas.drawPath(stroke, paint)
        }
        canvas.drawPath(currentStroke, paint)
    }

    data class PathData(val points: List<Triple<Float, Float, Long>>)

    fun getAllPoints(): List<List<Triple<Float, Float, Long>>> {
        return paths.map { it.points }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y
        val timestamp = SystemClock.uptimeMillis().toFloat()

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                path.moveTo(x, y)
                currentStroke.moveTo(x, y)
                lastX = x
                lastY = y
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val controlX = (lastX + x) / 2
                val controlY = (lastY + y) / 2
                currentStroke.quadTo(lastX, lastY, controlX, controlY)
                lastX = x
                lastY = y
                invalidate()
            }
            MotionEvent.ACTION_UP -> {
                currentStroke.lineTo(x, y)
                strokes.add(currentStroke)
                paths.add(PathData(currentStroke.asPoints()))
                currentStroke = Path()
                invalidate()
            }
        }
        return true
    }

    private fun Path.asPoints(): List<Triple<Float, Float, Long>> {
        val points = mutableListOf<Triple<Float, Float, Long>>()
        val pathMeasure = android.graphics.PathMeasure(this, false)
        val pathLength = pathMeasure.length
        val interval = 5f
        var distance = 0f
        val coordinates = floatArrayOf(0f, 0f)

        while (distance < pathLength) {
            pathMeasure.getPosTan(distance, coordinates, null)
            points.add(Triple(coordinates[0], coordinates[1], SystemClock.uptimeMillis()))
            distance += interval
        }
        return points
    }

    fun clearCanvas() {
        path.reset()
        strokes.clear()
        paths.clear()
        invalidate()
    }

    data class PointData(val x: Float, val y: Float, val time: Float)
}
