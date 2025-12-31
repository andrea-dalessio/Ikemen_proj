package main

import (
	"fmt"
	"strings"
)

// Helper per premere/rilasciare tasti tramite il motore interno
// Assumiamo che OnKeyPressed/OnKeyReleased siano disponibili nel package main
func SimKey(key int, press bool) {
	if press {
		OnKeyPressed(Key(key), 0)
	} else {
		OnKeyReleased(Key(key), 0)
	}
}

// Questa funzione preme i tasti fisici giusti in base alla configurazione caricata
func ApplyNetworkInput(action AgentAction, p1Facing float32, p2Facing float32) {
	// Se sys.keyConfig è vuoto o non ha abbastanza giocatori, esci
	if len(sys.keyConfig) < 2 {
		return
	}

	// --- PLAYER 1 (Indice 0) ---
	processPlayerInput(0, action.P1Move, action.P1Btn, p1Facing)

	// --- PLAYER 2 (Indice 1) ---
	processPlayerInput(1, action.P2Move, action.P2Btn, p2Facing)
}

func processPlayerInput(playerIdx int, move string, btn string, facing float32) {
	// Ottieni la configurazione tasti per questo giocatore
	// sys.keyConfig deve essere di tipo []*KeyConfig o []KeyConfig
	cfg := sys.keyConfig[playerIdx]

	if move != "" || btn != "" {
		fmt.Printf("Player%d <Move:%s,Button:%s>\n", playerIdx, move, btn)
	}

	// 1. Decodifica Movimento (Relativo -> Assoluto)
	var up, down, left, right bool

	// 1. Verticali (Assoluti)
	if strings.Contains(move, "up") {
		up = true
	}
	if strings.Contains(move, "down") {
		down = true
	}

	// 2. Orizzontali (Relativi al Facing)
	// Se invii "F" (o "DF"), capisce dove guardi e preme la freccia giusta
	if strings.Contains(move, "forward") {
		if facing > 0 {
			right = true // Guarda a destra -> Premi Destra
		} else {
			left = true // Guarda a sinistra -> Premi Sinistra
		}
	}

	if strings.Contains(move, "back") {
		if facing > 0 {
			left = true // Guarda a destra -> Indietro è Sinistra
		} else {
			right = true // Guarda a sinistra -> Indietro è Destra
		}
	}

	// 2. Premi i Tasti Direzionali
	// Qui usiamo i nomi corretti della tua struct KeyConfig (dU, dD, dL, dR)
	SimKey(cfg.dU, up)
	SimKey(cfg.dD, down)
	SimKey(cfg.dL, left)
	SimKey(cfg.dR, right)

	// 3. Premi i Tasti Attacco
	// Usiamo strings.Contains per permettere combo di tasti (es. btn="ab" per EX Move)
	// Nota: Assicurati di aver importato "strings" in alto nel file.

	SimKey(cfg.kA, strings.Contains(btn, "a"))
	SimKey(cfg.kB, strings.Contains(btn, "b"))
	SimKey(cfg.kC, strings.Contains(btn, "c"))

	SimKey(cfg.kX, strings.Contains(btn, "x"))
	SimKey(cfg.kY, strings.Contains(btn, "y"))
	SimKey(cfg.kZ, strings.Contains(btn, "z"))

	SimKey(cfg.kS, strings.Contains(btn, "s")) // Start
}
