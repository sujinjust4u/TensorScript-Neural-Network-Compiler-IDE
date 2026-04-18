model Classifier {
  input shape(784)
  layer dense(128, relu)
  layer dropout(0.0)
  layer dropout(0.3)
  layer dense(10, softmax)
  loss categorical_crossentropy
  optimizer adam(lr=0.001)
  train epochs=5, batch=32
}
