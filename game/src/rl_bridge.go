package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"
)

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
	Reset  bool   `json:"reset"`
}

var (
	conn        net.Conn
	listener    net.Listener
	mu          sync.Mutex
	IsConnected bool = false
	netReader   *bufio.Reader
)

func StartRLServer() {
	fmt.Println("--- RL SERVER: Avvio RAW su porta 8080 ---")
	ln, err := net.Listen("tcp", ":8080")
	if err != nil {
		fmt.Println("Errore bind:", err)
		return
	}
	listener = ln

	go func() {
		for {
			c, err := ln.Accept()
			if err != nil { continue }
			
			mu.Lock()
			if conn != nil { conn.Close() }
			conn = c
			// TCP No Delay non è accessibile facilmente da net.Conn standard, ma bufio aiuta
			netReader = bufio.NewReader(conn)
			IsConnected = true
			mu.Unlock()
			
			fmt.Println("--- PYTHON CONNESSO (RAW MODE) ---")
		}
	}()
}

func SyncWithPython(state RLGameState) AgentAction {
	mu.Lock()
	defer mu.Unlock()

	if !IsConnected || conn == nil {
		return AgentAction{}
	}

	// 1. SERIALIZZA JSON (Senza encoder bufferizzato)
	bytes, err := json.Marshal(state)
	if err != nil {
		return AgentAction{}
	}

	// 2. INVIA JSON + NEWLINE (Cruciale!)
	// Impostiamo una deadline per evitare freeze eterni
	conn.SetWriteDeadline(time.Now().Add(2 * time.Second))
	_, err = conn.Write(append(bytes, '\n'))
	if err != nil {
		fmt.Println("Write Error (Resetting connection):", err)
		conn.Close()
		IsConnected = false
		return AgentAction{}
	}

	// 3. RICEVI RISPOSTA (Bloccante con Timeout)
	conn.SetReadDeadline(time.Now().Add(60 * time.Second)) // 5 secondi max per rispondere
	line, err := netReader.ReadString('\n')
	if err != nil {
		fmt.Println("Read Error (Timeout o Disconnessione):", err)
		conn.Close()
		IsConnected = false
		return AgentAction{}
	}

	// 4. DESERIALIZZA
	line = strings.TrimSpace(line)
	var action AgentAction
	if len(line) > 0 {
		err = json.Unmarshal([]byte(line), &action)
		if err != nil {
			fmt.Println("JSON Error:", err)
		}
	}

	return action
}
