import {
  useRef,
  useEffect,
  useState,
  useImperativeHandle,
  forwardRef,
  useCallback,
} from "react";

export interface SketchCanvasRef {
  clear: () => void;
  exportBlob: () => Promise<Blob | null>;
}

interface SketchCanvasProps {
  width: number;
  height: number;
  strokeWidth: number;
  tool: "pen" | "eraser";
}

export const SketchCanvas = forwardRef<SketchCanvasRef, SketchCanvasProps>(
  function SketchCanvas({ width, height, strokeWidth, tool }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const lastPos = useRef<{ x: number; y: number } | null>(null);

    // Initialize canvas with white background
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, height);
    }, [width, height]);

    const getCanvasCoords = useCallback(
      (e: React.MouseEvent | React.TouchEvent) => {
        const canvas = canvasRef.current;
        if (!canvas) return null;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        let clientX: number;
        let clientY: number;

        if ("touches" in e) {
          const touch = e.touches[0] || e.changedTouches[0];
          if (!touch) return null;
          clientX = touch.clientX;
          clientY = touch.clientY;
        } else {
          clientX = e.clientX;
          clientY = e.clientY;
        }

        return {
          x: (clientX - rect.left) * scaleX,
          y: (clientY - rect.top) * scaleY,
        };
      },
      [],
    );

    const startDrawing = useCallback(
      (e: React.MouseEvent | React.TouchEvent) => {
        e.preventDefault();
        const coords = getCanvasCoords(e);
        if (!coords) return;

        setIsDrawing(true);
        lastPos.current = coords;
      },
      [getCanvasCoords],
    );

    const draw = useCallback(
      (e: React.MouseEvent | React.TouchEvent) => {
        if (!isDrawing || !lastPos.current) return;
        e.preventDefault();

        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) return;

        const coords = getCanvasCoords(e);
        if (!coords) return;

        ctx.beginPath();
        ctx.moveTo(lastPos.current.x, lastPos.current.y);
        ctx.lineTo(coords.x, coords.y);
        ctx.strokeStyle = tool === "eraser" ? "white" : "black";
        ctx.lineWidth = strokeWidth;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();

        lastPos.current = coords;
      },
      [isDrawing, tool, strokeWidth, getCanvasCoords],
    );

    const stopDrawing = useCallback(() => {
      setIsDrawing(false);
      lastPos.current = null;
    }, []);

    useImperativeHandle(ref, () => ({
      clear: () => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) return;

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);
      },
      exportBlob: () => {
        return new Promise((resolve) => {
          const canvas = canvasRef.current;
          if (!canvas) {
            resolve(null);
            return;
          }
          canvas.toBlob((blob) => resolve(blob), "image/png");
        });
      },
    }));

    return (
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="border rounded-lg bg-white cursor-crosshair touch-none"
        style={{ maxWidth: "100%", height: "auto" }}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
      />
    );
  },
);
