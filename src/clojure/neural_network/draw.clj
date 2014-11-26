(ns neural-network.draw)
(use 'neural-network.utils)

(import java.awt.Color)
(import java.awt.image.BufferedImage)
(import java.io.File)
(import java.io.IOException)
(import javax.imageio.ImageIO)

;;Stuff to draw .png files

(defn drawer [height width]
  "Returns a drawing function for an image with height and width specified by the corresponding parameters.
  A drawing function takes either a hue, brightness, and [y x] coordinate pair, or a filename string.
  In the first case, it draws a pixel with the corresponding hue and brightness at the specified location.
  In the second case, it saves the image it has drawn to the given filename."
  (let [pixelImage (BufferedImage. width height (BufferedImage/TYPE_INT_RGB))]
    (fn ([hue brightness [y x]]
         (.setRGB pixelImage x y (Color/HSBtoRGB hue 1.0 brightness)))
      ([filename]
       (ImageIO/write pixelImage "png" (File. filename))))))

(defn draw-vect [v height width filename]
  "Draws a two-dimensional representation of the vector la4j vector v."
  (let [max (-> v vect-abs .max)
        d (drawer height width)]
    (dovect v #(d (if (pos? %2) 0.5 0.0)
                  (/ (Math/abs %2) max)
                  (coords %1 width)))
    (d filename)))
