# ChessGPT4
Chess AI for ECE4318 Project


Demo in classPT2
```mermaid
sequenceDiagram
    autonumber
    actor Host as Host/ GUI (Arena)
    participant Engine as Engine
    participant Parser as UCI Parser
    participant Position as Poisition (Board/FEN)
    participant MoveGen as MoveGenerator (pseudoLegalMoves etc.)
    participant Rule as RuleChecker (isSquareAttacked / inCheck)
    participant MoveObj as Move (Move.fromUci / toUci)

    
    Host ->> Engine: "uci"
    activate Engine
    Engine ->> Engine: parse "uci"
    Engine -->> Host: "id name <engine>"
    Engine -->> Host: "id author <author>"
    Engine -->> Host: "uciok"
    deactivate Engine
    Host -->> Engine: "isready"
    activate Engine
    Engine --> Engine: parse "isready"
    Engine --> Host: "readyok"
    deactivate Engine

    Host ->> Engine: "ucinewgame"
    activate Engine
    Engine ->> Position: Position.startPos() // reset position to starting
    Engine -->> Host: (ack no response required)
    deactivate Engine

    Host ->> Engine: "position startpos moves e2e4 e7e5 ..."
    activate Engine
    Engine ->> Parser: parse "position ..." (detect startpos (or fen))
    Parser ->> Position: create Position (startPos (or fromFEN))
    loop for each move token
        Parser ->> MoveObj: Move.fromUci("e2e4")
        MoveObj ->> Position: Position = Position = Position.makeMove(Move)
    end
    Engine -->> Host: (no response required)
    deactivate Engine

```
