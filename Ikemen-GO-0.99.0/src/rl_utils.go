package main

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

	// 1. Decodifica Movimento (Relativo -> Assoluto)
	var up, down, left, right bool

	if move == "U" {
		up = true
	}
	if move == "D" {
		down = true
	}

	// Logica per Avanti (F) e Indietro (B) basata sulla direzione del personaggio
	if move == "F" {
		if facing > 0 {
			right = true
		} else {
			left = true
		}
	}
	if move == "B" {
		if facing > 0 {
			left = true
		} else {
			right = true
		}
	}

	// 2. Premi i Tasti Direzionali
	// Qui usiamo i nomi corretti della tua struct KeyConfig (dU, dD, dL, dR)
	SimKey(cfg.dU, up)
	SimKey(cfg.dD, down)
	SimKey(cfg.dL, left)
	SimKey(cfg.dR, right)

	// 3. Premi i Tasti Attacco
	// Qui usiamo i nomi corretti della tua struct KeyConfig (kA, kB, ecc.)
	SimKey(cfg.kA, btn == "a")
	SimKey(cfg.kB, btn == "b")
	SimKey(cfg.kC, btn == "c")
	SimKey(cfg.kX, btn == "x")
	SimKey(cfg.kY, btn == "y")
	SimKey(cfg.kZ, btn == "z")

	// Start (S)
	SimKey(cfg.kS, btn == "s")
}
