package main

// --- COSTANTI DI INPUT ---
// 1. Definiamo le DIREZIONI (perché NON esistono in input.go)
//    Dobbiamo usare il tipo InputBits per compatibilità.
const (
	IB_UP    InputBits = 1 << 0
	IB_DOWN  InputBits = 1 << 1
	IB_LEFT  InputBits = 1 << 2
	IB_RIGHT InputBits = 1 << 3
)

// 2. I TASTI (IB_A, IB_B, ecc.) esistono già in input.go.
//    NON li ridefiniamo qui per evitare errori di "redeclared".

// MODIFICA: La funzione ora accetta stringhe generiche (moveStr, btnStr)
// Invece di prendere l'intera struct AgentAction.
func ConvertToInput(moveStr string, btnStr string, facing float32) InputBits {
	var bits InputBits = 0

	// 1. INPUT DIREZIONALE
	switch moveStr {
	case "U":
		bits |= IB_UP
	case "D":
		bits |= IB_DOWN
	case "F": // Forward (Avanti relativo)
		if facing > 0 {
			bits |= IB_RIGHT
		} else {
			bits |= IB_LEFT
		}
	case "B": // Back (Indietro relativo)
		if facing > 0 {
			bits |= IB_LEFT
		} else {
			bits |= IB_RIGHT
		}
	}

	// 2. INPUT TASTI
	switch btnStr {
	case "a":
		bits |= IB_A
	case "b":
		bits |= IB_B
	case "c":
		bits |= IB_C
	case "x":
		bits |= IB_X
	case "y":
		bits |= IB_Y
	case "z":
		bits |= IB_Z
	case "s":
		bits |= IB_S
	}

	return bits
}
