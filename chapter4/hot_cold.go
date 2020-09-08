package main

import (
  "fmt"
  "math"
)

func main() {
  weight := 0.1
  lr := 0.01

  numberOfToes := [...]float64{8.5}
  winOrLoseBinary := [...]float64{1.0}  // won!

  input := numberOfToes[0]
  won := winOrLoseBinary[0]

  prediction := neuralNetwork(input, weight)
  fmt.Println("Prediction ", prediction)

  err := math.Pow((prediction - float64(won)), 2)
  fmt.Println("Error ", err)

  pUp := neuralNetwork(input, weight + lr)
  eUp := math.Pow((pUp - won), 2)

  fmt.Println("Higher weight error ", eUp)

  pDn := neuralNetwork(input, weight - lr)
  eDn := math.Pow((pDn - won), 2)

  fmt.Println("Lower weight error ", eDn)

  if err > eDn || err > eUp {
    if eDn < eUp {
      weight -= lr
    }
    if eUp < eDn {
      weight += lr
    }
  }

  fmt.Println("Final weight ", weight)
}

func neuralNetwork(input, weight float64) float64 {
  return input * weight
}
