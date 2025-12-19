package main

import (
	"encoding/json"
	"fmt"
	"net"
	"sync"
)

// Definiamo una struct SPECIFICA per l'RL, con un nome diverso da "GameState"
type RLGameState struct {
    P1_HP      int32   `json:"p1_hp"`
    P1_X       float32 `json:"p1_x"`
    P1_Y       float32 `json:"p1_y"`
    P1_Power   int32   `json:"p1_power"`
    P1_LifeMax int32   `json:"p1_life_max"`
    P1_Facing  float32 `json:"p1_facing"`
    P1_AnimNo  int32   `json:"p1_anim_no"`

    P2_HP      int32   `json:"p2_hp"`
    P2_X       float32 `json:"p2_x"`
    P2_Y       float32 `json:"p2_y"`
    P2_Power   int32   `json:"p2_power"`
    P2_LifeMax int32   `json:"p2_life_max"`
    P2_Facing  float32 `json:"p2_facing"`
    P2_AnimNo  int32   `json:"p2_anim_no"`

    GameTick int `json:"tick"`
}

type AgentAction struct {
	P1Move string `json:"p1_move"`
	P1Btn  string `json:"p1_btn"`
	P2Move string `json:"p2_move"`
	P2Btn  string `json:"p2_btn"`
	Reset   bool   `json:"reset"`
}

var (
	conn        net.Conn
	listener    net.Listener
	mu          sync.Mutex
	IsConnected bool = false
)

func StartRLServer() {
	fmt.Println("--- RL SERVER: In attesa di connessione Python su porta 8080 ---")
	ln, err := net.Listen("tcp", ":8080")
	if err != nil {
		fmt.Println("Errore avvio server:", err)
		return
	}
	listener = ln
	
	for {
		c, err := ln.Accept()
		if err != nil {
			fmt.Println("Errore connessione:", err)
			continue
		}
		conn = c
		IsConnected = true
		fmt.Println("--- RL SERVER: Python Connesso! ---")
	}
}

// Aggiorniamo la firma per usare RLGameState
func SyncWithPython(state RLGameState) AgentAction {
	if !IsConnected {
		return AgentAction{}
	}

	encoder := json.NewEncoder(conn)
	err := encoder.Encode(state)
	if err != nil {
		fmt.Println("Errore invio stato:", err)
		IsConnected = false
		return AgentAction{}
	}

	var action AgentAction
	decoder := json.NewDecoder(conn)
	err = decoder.Decode(&action)
	if err != nil {
		fmt.Println("Errore ricezione azione:", err)
		IsConnected = false
		return AgentAction{}
	}

	return action
}
