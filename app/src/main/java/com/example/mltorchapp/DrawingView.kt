package com.example.mltorchapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class DrawingView(context: Context, attrs: AttributeSet) : View(context, attrs) {
    private var paint: Paint = Paint()
    private var path: Path = Path()
    private var allPoints = mutableListOf<MutableList<Triple<Float, Float, Long>>>()
    private var consecutivePoints = mutableListOf<Triple<Float, Float, Long>>()
    private var maxX = 0f
    private var maxY = 0f

    init {
        paint.apply {
            color = Color.BLACK
            style = Paint.Style.STROKE
            strokeJoin = Paint.Join.ROUND
            strokeCap = Paint.Cap.ROUND
            strokeWidth = 10f
            isAntiAlias = true
        }
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        maxX = w.toFloat()
        maxY = h.toFloat()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawPath(path, paint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = maxY - event.y  // Adjust coordinate system to match bottom-left origin
        val normalizedX = x / maxX
        val normalizedY = y / maxY
        val timestamp = System.currentTimeMillis()

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                consecutivePoints.clear()
                consecutivePoints.add(Triple(normalizedX, normalizedY, timestamp))
                path.moveTo(x, event.y)  // Use event.y directly for drawing
                invalidate()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                consecutivePoints.add(Triple(normalizedX, normalizedY, timestamp))
                path.lineTo(x, event.y)  // Use event.y directly for drawing
                invalidate()
            }
            MotionEvent.ACTION_UP -> {
                consecutivePoints.add(Triple(normalizedX, normalizedY, timestamp))
                allPoints.add(ArrayList(consecutivePoints))
            }
        }
        invalidate()
        return true
    }

    fun clearCanvas() {
        path.reset()
        invalidate()
        allPoints.clear()
    }

    fun getAllPoints(): List<List<Triple<Float, Float, Long>>> {
        return allPoints
    }
}
