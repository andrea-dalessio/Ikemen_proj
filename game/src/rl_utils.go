package main

import (
	"fmt"
	"strings"
)

// Codici Tasti GLFW (Standard per Ikemen/MUGEN)
// Se P1 Verticale funziona, questi codici sono corretti.
const (
	// Player 1 Defaults
	KEY_UP    = 265
	KEY_DOWN  = 264
	KEY_LEFT  = 263
	KEY_RIGHT = 262
	KEY_Z     = 90 // A
	KEY_X     = 88 // B
	KEY_C     = 67 // C
	KEY_A     = 65 // X
	KEY_S     = 83 // Y
	KEY_D     = 68 // Z
	KEY_ENTER = 257 // Start

	// Player 2 Defaults
	KEY_I     = 73 
	KEY_K     = 75 
	KEY_J     = 74 
	KEY_L     = 76 
	KEY_Q     = 81 
	KEY_W     = 87 
	KEY_E     = 69 
	KEY_R     = 82 
	KEY_T     = 84 
	KEY_Y     = 89 
	KEY_U     = 85 
)

func SimKey(key int, press bool) {
	if press {
		OnKeyPressed(Key(key), 0)
	} else {
		OnKeyReleased(Key(key), 0)
	}
}

func ApplyNetworkInput(action AgentAction, p1Facing float32, p2Facing float32) {
	if len(sys.keyConfig) < 2 { return }
	processPlayerInput(0, action.P1Move, action.P1Btn, p1Facing)
	processPlayerInput(1, action.P2Move, action.P2Btn, p2Facing)
}

func processPlayerInput(playerIdx int, move string, btn string, facing float32) {
	// Definizione tasti
	var kUp, kDown, kLeft, kRight, kA, kB, kC, kX, kY, kZ, kStart int

	if playerIdx == 0 {
		kUp, kDown, kLeft, kRight = KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT
		kA, kB, kC = KEY_Z, KEY_X, KEY_C
		kX, kY, kZ = KEY_A, KEY_S, KEY_D
		kStart = KEY_ENTER
	} else {
		kUp, kDown, kLeft, kRight = KEY_I, KEY_K, KEY_J, KEY_L
		kA, kB, kC = KEY_Q, KEY_W, KEY_E
		kX, kY, kZ = KEY_R, KEY_T, KEY_Y
		kStart = KEY_U
	}

	// --- 1. DECODIFICA MOVIMENTO ---
	var up, down, left, right bool

	// A. DIREZIONI ASSOLUTE (Se Python invia "up left", "right", ecc.)
	// Usiamo Contains per catturare anche diagonali inviate come "up left"
	if strings.Contains(move, "up") || strings.Contains(move, "U") { up = true }
	if strings.Contains(move, "down") || strings.Contains(move, "D") { down = true }
	
	// Nota: controlliamo "left" minuscolo per evitare confusione con "L" di Player 2 tasto L se mai ci fosse ambiguità
	if strings.Contains(move, "left") { left = true }
	if strings.Contains(move, "right") { right = true }
	
	// Gestione codici brevi assoluti "L" e "R" (solo se stringa esatta o parte di diagonale nota)
	if strings.Contains(move, "L") && !strings.Contains(move, "Left") { left = true }
	if strings.Contains(move, "R") && !strings.Contains(move, "Right") { right = true }

	// B. DIREZIONI RELATIVE (Se Python invia "F", "UF", "B", "DB")
	// Fallback se non abbiamo già deciso L/R
	if !left && !right {
		// FIX: Usa Contains invece di == per catturare "UF", "DF"
		isForward := strings.Contains(move, "F") || strings.Contains(move, "forward")
		isBack := strings.Contains(move, "B") || strings.Contains(move, "back")

		if isForward {
			if facing > 0 { right = true } else { left = true }
		} else if isBack {
			if facing > 0 { left = true } else { right = true }
		}
	}

	// --- DEBUG PRINT DIRETTO ---
	// Decommenta questo se ancora non funziona per vedere cosa decide Go
	if move != "" {
		fmt.Printf("[P%d] In: '%s' (Fac:%.0f) -> U:%v D:%v L:%v R:%v\n", playerIdx, move, facing, up, down, left, right)
	}
	
	// --- 2. ESECUZIONE ---
	SimKey(kUp, up)
	SimKey(kDown, down)
	SimKey(kLeft, left)
	SimKey(kRight, right)

	SimKey(kA, strings.Contains(btn, "a"))
	SimKey(kB, strings.Contains(btn, "b"))
	SimKey(kC, strings.Contains(btn, "c"))
	SimKey(kX, strings.Contains(btn, "x"))
	SimKey(kY, strings.Contains(btn, "y"))
	SimKey(kZ, strings.Contains(btn, "z"))
	SimKey(kStart, strings.Contains(btn, "s"))
}