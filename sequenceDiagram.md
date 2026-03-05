```mermaid

sequenceDiagram
    autonumber
    
    %% Define Actors and Participants
    actor Host as Host / GUI (Arena)
    participant Engine as Engine (Main loop)
    participant Parser as UCI Parser
    participant Position as Position (Board/FEN)
    participant MoveGen as MoveGenerator
    participant Rule as RuleChecker
    participant MoveObj as Move (Move Object)

    %% Initialization Phase
    Note over Host, Engine: Initialization
    Host ->> Engine: "uci"
    activate Engine
    Engine ->> Engine: parse "uci"
    Engine -->> Host: "id name <engine>"
    Engine -->> Host: "id author <author>"
    Engine -->> Host: "uciok"
    deactivate Engine

    Host ->> Engine: "isready"
    activate Engine
    Engine ->> Engine: parse "isready"
    Engine -->> Host: "readyok"
    deactivate Engine

    %% New Game and Position Setup
    Note over Host, Position: Setup Board
    Host ->> Engine: "ucinewgame"
    activate Engine
    Engine ->> Position: Position.startPos()
    Engine -->> Host: (ack)
    deactivate Engine

    Host ->> Engine: "position startpos moves e2e4"
    activate Engine
    Engine ->> Parser: parse "position..."
    Parser ->> Position: create Position(startPos)
    
    loop for each move token
        Parser ->> MoveObj: Move.fromUci("e2e4")
        MoveObj -->> Parser: Move object
        Parser ->> Position: Position.makeMove(Move)
    end
    deactivate Engine


%% Move Generation with Rule Checker Logic [cite: 147, 154, 157, 160]
    Note over Host, Rule: Thinking & Rule Checking
    Host ->> Engine: "go movetime 10000"
    activate Engine
    Engine ->> Parser: parse "go" options

    Engine ->> MoveGen: legalMoves = Position.legalMoves()
    activate MoveGen

    loop for each pseudo-legal move
        MoveGen ->> Rule: isSquareAttacked(kingSquare)
        activate Rule
        Rule -->> MoveGen: boolean (inCheck)
        deactivate Rule

        Note right of MoveGen: If move is valid, add to list
    end

    MoveGen -->> Engine: returns legalMoves list [cite: 147, 161]
    deactivate MoveGen


%% Selecting and Sending Move
    Engine ->> Engine: choose best move
    Engine ->> MoveObj: selectedMove.toUci()
    Engine -->> Host: "bestmove e2e4"
    deactivate Engine

    %% Shutdown
    Host ->> Engine: "quit"
    activate Engine
    Engine ->> Engine: cleanup and exit
    Engine -->> Host: (process ends)
    deactivate Engine
```
