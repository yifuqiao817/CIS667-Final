import yqiao08_Milestone1 as ms1 #Import the chess states
import yqiao08_Milestone2 as ms2 #Import the AI settings
import random as rd
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import numpy as np
import torch as tr
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Sequential, Linear, Flatten, Tanh


## Encode the states into network inputs
def chessboard_encoding(board, game_size, side): #game_size: N, M0, M1, M2, M3
    size = np.shape(board)[0]
    if(size == 4):
        num_piece = 8
    elif(size == 8):
        num_piece = 32
    encoded_state = np.zeros((num_piece, size, size))
    if(game_size[0] == 'N'):
        chessnames = ms1.White_chess + ms1.Black_chess
    elif(game_size[0] == 'M'):
        serial = int(game_size[1])
        chessnames = ms1.White_chess_44[serial] + ms1.Black_chess_44[serial]
    for row in range(size):
        for col in range(size):
            if(str(board[row,col], encoding='utf-8') in chessnames):
                if(str(board[row,col], encoding='utf-8')[0] == side):
                    flag = 1
                else:
                    flag = -1
                index = chessnames.index(str(board[row,col],encoding='utf-8'))
                encoded_state[index, row, col] = flag * 1
    return encoded_state

class AI_choices_data(object): # Unlike that in Milestone2, this one is only used for training data.
    def __init__(self,chess,color):
        self.chessstate = chess #The state of the chessboard
        self.side = color #The side which AI plays
        self.score = 0 #The score of the node in Minimax algorithm
        self.children = [] #Children of the node in the tree
        self.move = (0,0,0) #How did the parent move to the current node
                            #(Piece, Target, 0) for moving or (Piece1, Piece2, 1) for swapping
    def copy(self):
        return ms2.AI_choices(chess=self.chessstate, color=self.side)

    def minimax(self, masterside): #Optimize the option
        if(len(self.children) == 0): #When the node is leaf
            self.score = self.score_current() #Calculate the score according to the state
        else:
            candidates = []
            for child in self.children:
                candidates.append(child.minimax(masterside)) #All the scores for children
            if (self.side == masterside):  # AI's turn
                self.score = max(candidates)
            else: #Opponent's turn
                self.score = min(candidates)
        return self.score


    def newstate(self, state):
        self.chessstate = state

    def possible_options(self): #Find all the possible options
        nodes = []
        for piece in self.chessstate.pieces: #For moving
            if (piece.side == self.side):
                for row in self.chessstate.rows:
                    for col in self.chessstate.cols:
                        if(col+row != piece.curr_pos):
                            if(self.chessstate.fake_move_for_AI(piece,col+row) == 1): #Valid option
                                nodes.append((piece.name,col+row,0))
        for piece1 in self.chessstate.pieces: #For swapping
            if(piece1.side == self.side):
                for piece2 in self.chessstate.pieces:
                    if((piece2.side == self.side) and (piece1.name != piece2.name)): #Valid option
                        if((piece2.name, piece1.name) not in nodes):
                            nodes.append((piece1.name,piece2.name,1))
        return nodes

    def score_current(self): #Calculate the score according to state
        return self.chessstate.get_score(self.side)

    def score_evaluation(self,command,type):
        temp_state = copy.deepcopy(self.chessstate)
        if(type == 0):
            temp_state.one_move(command[0],command[1])
        elif(type == 1):
            temp_state.one_swap(command[0],command[1])
        return temp_state.get_score(self.side)

def opposite_side(side): #Get the opposite side.
    if(side == 'W'):
        return 'B'
    if(side == 'B'):
        return 'W'

class treebased_AI_data(object): #Tree based AI, only for generating training data.
    def __init__(self, chess, color):
        self.AI = AI_choices_data(chess,color)

    def renew(self, new_state): #Update the new state
        self.AI.newstate(new_state)

    def make_option(self):
        #Based on Minimax and have a max depth
        max_depth = 2 #Enough when against baseline AI: take pieces as many as possible
        i = 0
        temp_AI = copy.deepcopy(self.AI) #Copy the root node
        nodes = [[temp_AI]] #List of nodes
        nodes_count = 1
        while(i < max_depth):
            current_nodes = nodes[i] #i depth
            new_nodes = [] #i+1 depth
            for one_node in current_nodes:
                current_side = one_node.side
                possible_options = one_node.possible_options()
                for op in possible_options:
                    current_state = copy.deepcopy(one_node.chessstate)
                    if(op[2] == 0): #Moving option
                        current_state.one_move(op[0], op[1])
                        new_AI = AI_choices_data(chess=current_state, color=opposite_side(current_side))
                        new_AI.move = op
                    elif(op[2] == 1): #Swapping option
                        current_state.one_swap(op[0], op[1])
                        new_AI = AI_choices_data(chess=current_state, color=opposite_side(current_side))
                        new_AI.move = op
                    one_node.children.append(copy.copy(new_AI))
                    new_nodes.append(copy.copy(new_AI)) #Not deep copy for we want to get access to the children
            nodes.append(new_nodes)
            nodes_count += len(new_nodes)
            i += 1
        print('There are ' + str(nodes_count) + ' nodes for the tree to process')
        master_side = temp_AI.side
        final_score = nodes[0][0].minimax(master_side)
        candidate_count = 0
        data_batch = []
        selections = []
        for child in nodes[0][0].children: #Choose the next step
            stateboard = chessboard_encoding(child.chessstate.chessboard, child.chessstate.size + str(child.chessstate.minitype), master_side)
            data_batch.append((stateboard,child.score)) #Load the data for each child
            if(child.score == final_score):
                candidate_count += 1
                #one_selection = child.move
                selections.append(child.move)
        rd.shuffle(selections)
        if(len(selections) == 1):
            one_selection = selections[0]
        else:
            option_choice = rd.randint(0, 1)
            option_another = abs(1 - option_choice)
            for op in [option_choice, option_another]:
                for option in selections:
                    if (option[-1] == op):
                        one_selection = option
        print(one_selection)
        if (one_selection[2] == 0):
            self.AI.chessstate.one_move(one_selection[0], one_selection[1])
        elif (one_selection[2] == 1):
            self.AI.chessstate.one_swap(one_selection[0], one_selection[1])
        return data_batch #The data collected

    def show_state(self):
        self.AI.chessstate.show_state()

    def checkmate(self, whosturn): #If one took the king, it wins
        kingname = whosturn + 'K'
        if(self.find_chess(kingname) == 0):
            return 1
        else:
            return 0

    def find_chess(self, target):
        return self.AI.chessstate.find_chess(target)

def AI_match_forData(): #Tree based AI vs Tree based AI for data generating
    for problem_size in [1,2,3,4,5]:
        winners = []
        net_data = [] #The data to be stored, one for each problem size
        for game_num in range(10):
            if (problem_size == 1):
                mini_select = 'False' #8*8 game
            else:
                mini_select = 'True' #4*4 game
            new_game = ms1.chess_state(mini=mini_select, minitype=problem_size)
            player1 = treebased_AI_data(new_game, 'W')
            player2 = treebased_AI_data(new_game, 'B')
            turn = 0
            while (turn >= 0):
                print(problem_size,turn,winners)
                print('So far collected: ',len(net_data))
                if (turn % 2 == 0):
                    databatch = player1.make_option()
                    new_state = player1.AI.chessstate
                    player2.renew(new_state) #We have to update game state for each AI once per turn
                else:
                    databatch = player2.make_option()
                    new_state = player2.AI.chessstate
                    player1.renew(new_state)
                net_data.extend(databatch) #Update the data
                new_state.show_state()
                turn += 1
                if(player1.checkmate('B') == 1): #White wins
                    winners.append('W')
                    break
                if(player2.checkmate('W') == 1): #Black wins
                    winners.append('B')
                    break
                if(turn == 50): #Time limit
                    current_state = player1.AI.chessstate
                    if(current_state.get_score('W') > current_state.get_score('B')):
                        winners.append('W')
                    elif(current_state.get_score('W') < current_state.get_score('B')):
                        winners.append('B')
                    else:
                        winners.append('Draw')
                    break
        print(winners)
        with open('C:/Users/28529/Desktop/SU/Courses/AI/project/new code/netdata_' + str(problem_size) + '.pkl','wb') as fp:
            pickle.dump(net_data, fp, pickle.HIGHEST_PROTOCOL) #Store data into pickle files
        del net_data

class chess_net(nn.Module): #The network model
    def __init__(self, problem_size):
        super(chess_net,self).__init__()
        if (problem_size[0] == 'N'):
            board_size = 8
            piece_size = 32
        else:
            board_size = 4
            piece_size = 8
        self.board_size = board_size
        self.piece_size = piece_size
        self.fc1 = Linear(in_features=piece_size*board_size*board_size, out_features=128, bias=True)
        self.fc2 = Linear(in_features=128, out_features=1, bias=True) #Two fully-connected layers

    def forward(self, x):
        x_flattened = x.reshape((x.shape[0], self.piece_size*self.board_size*self.board_size))
        out = tr.tanh(self.fc1(x_flattened)) #Tanh activation function
        out = self.fc2(out)
        return out

def calculate_loss(net, x, y_targ): #Calculate loss value
    y_out = net(x)
    errors = tr.mean((y_out - y_targ) ** 2)
    return (y_out, errors)

def optimization_step(optimizer, net, x, y_targ): #Optimize the network parameters in each epoch
    optimizer.zero_grad()
    y_out, errors = calculate_loss(net,x,y_targ)
    errors.backward()
    optimizer.step()
    return (y_out, errors)


def model_training(problem_size, data): #Train network models
    threshold = 10000 #We got a lot of data, we need to sample
    rd.shuffle(data)
    data_unique = []
    replica = 0
    count = 0
    for new in data:
        count += 1
        flag = 1
        for old in data_unique:
            if((new[0] == old[0]).all() and (new[1] == old[1])): #Some (state, score) pairs are identical, we only need one
                replica += 1
                flag = 0
                break
        if(flag == 1):
            data_unique.append(new)
        print(count, len(data_unique), replica)
        if(len(data_unique) == threshold): #Extract the sample
            break
    rd.shuffle(data_unique)
    net = chess_net(problem_size=problem_size) #Initialize network
    X_data = []
    Y_data = []
    for item in data_unique:
        X_data.append(item[0])
        Y_data.append(item[1])
    X_data = np.array(X_data)
    print(np.shape(X_data))
    print(np.shape(Y_data))
    total_len = len(Y_data)
    train_size = int(0.8 * total_len)
    X_train = Variable(tr.tensor(X_data[:train_size, :, :, :],dtype=tr.float32))
    Y_train = Variable(tr.tensor(Y_data[:train_size],dtype=tr.float32))
    X_test = Variable(tr.tensor(X_data[train_size:, :, :, :],dtype=tr.float32))
    Y_test = Variable(tr.tensor(Y_data[train_size:],dtype=tr.float32))
    optimizer = tr.optim.Adam(net.parameters())
    y_axis_learning_train = []
    y_axis_learning_test = []
    for epoch in range(5000):
        ytrain_out, errors_train = optimization_step(optimizer,net, X_train, Y_train)
        ytest_out, erroes_test = calculate_loss(net, X_test, Y_test)
        if(epoch % 10 == 0):
            print(epoch, errors_train.item(), erroes_test.item())
            y_axis_learning_train.append(errors_train.item())
            y_axis_learning_test.append(erroes_test.item())
    tr.save(net.state_dict(), 'C:/Users/28529/Desktop/SU/Courses/AI/project/new code/model_'+ problem_size +'.pth') #Store network parameters
    plt.plot(y_axis_learning_train, 'b-')
    plt.plot(y_axis_learning_test, 'r-')
    plt.legend(["Train", "Test"])
    plt.xlabel("Iteration")
    plt.ylabel("Average Loss")
    plt.show() #Plot learning curves

    plt.plot(ytrain_out.detach().numpy(), Y_train.detach().numpy(), 'bo')
    plt.plot(ytest_out.detach().numpy(), Y_test.detach().numpy(), 'ro')
    plt.legend(["Train", "Test"])
    plt.xlabel("Actual output")
    plt.ylabel("Target output")
    plt.show() #Plot scatters


class NNbased_AI(object): #Network-based AI
    def __init__(self, chess, color):
        self.AI = ms2.AI_choices(chess,color)

    def renew(self, new_state): #Update the new state
        self.AI.newstate(new_state)

    def make_option(self,net):
        temp_AI = copy.deepcopy(self.AI) #Copy the root node
        current_side = temp_AI.side
        possible_options = temp_AI.possible_options()
        scores = []
        print('There are ' + str(len(possible_options)) + ' nodes for the tree to process')
        for op in possible_options:
            current_state = copy.deepcopy(temp_AI.chessstate)
            if (op[2] == 0):  # Moving option
                current_state.one_move(op[0], op[1])
                new_AI = ms2.AI_choices(chess=current_state, color=opposite_side(current_side))
                new_AI.move = op
            elif (op[2] == 1):  # Swapping option
                current_state.one_swap(op[0], op[1])
                new_AI = ms2.AI_choices(chess=current_state, color=opposite_side(current_side))
                new_AI.move = op
            encoded_state = chessboard_encoding(current_state.chessboard, current_state.size + str(current_state.minitype), temp_AI.side)
            encoded_state = Variable(tr.tensor(encoded_state,dtype=tr.float32)).reshape((1, encoded_state.shape[0], encoded_state.shape[1], encoded_state.shape[2]))
            predict_out = net(encoded_state).item()
            new_AI.score = predict_out
            scores.append(predict_out)
            temp_AI.children.append(copy.copy(new_AI))
        maxscore = np.max(scores)
        candidates = []
        for child in temp_AI.children:
            if(child.score == maxscore):
                candidates.append(child)
        if(len(candidates) == 1):
            one_selection = candidates[0].move
        else:
            rd.shuffle(candidates)
            one_selection = candidates[0].move
        print(one_selection)
        if (one_selection[2] == 0):
            self.AI.chessstate.one_move(one_selection[0], one_selection[1])
        elif (one_selection[2] == 1):
            self.AI.chessstate.one_swap(one_selection[0], one_selection[1])

    def show_state(self):
        self.AI.chessstate.show_state()

    def checkmate(self, whosturn): #If one took the king, it wins
        kingname = whosturn + 'K'
        if(self.find_chess(kingname) == 0):
            return 1
        else:
            return 0

    def find_chess(self, target):
        return self.AI.chessstate.find_chess(target)


def AI_match_forTest(): #Network AI vs Tree_based AI
    for problem_size in [1,2,3,4,5]: #Five problem sizes
        if(problem_size == 1):
            code = 'N'
        else:
            code = 'M' + str(problem_size-2)
        net = chess_net(code)
        net.load_state_dict(tr.load('C:/Users/28529/Desktop/SU/Courses/AI/project/new code/model_'+ code +'.pth'))  # Load the network
        winners = []
        scores = []
        for game_num in range(100):
            if (problem_size == 1):
                mini_select = 'False'  # 8*8 game
            else:
                mini_select = 'True'  # 4*4 game
            new_game = ms1.chess_state(mini=mini_select, minitype=problem_size)
            player1 = NNbased_AI(new_game, 'W')
            player2 = ms2.treebased_AI(new_game, 'B')
            turn = 0
            while (turn >= 0):
                print(problem_size, turn, winners)
                print(scores)
                if (turn % 2 == 0):
                    player1.make_option(net)
                    new_state = player1.AI.chessstate
                    player2.renew(new_state)  # We have to update game state for each AI once per turn
                else:
                    player2.make_option()
                    new_state = player2.AI.chessstate
                    player1.renew(new_state)
                new_state.show_state()
                turn += 1
                current_state = player1.AI.chessstate
                if (player1.checkmate('B') == 1):  # White wins
                    winners.append('W')
                    scores.append(current_state.get_score('W'))
                    break
                if (player2.checkmate('W') == 1):  # Black wins
                    winners.append('B')
                    scores.append(current_state.get_score('W'))
                    break
                if (turn == 50):  # Time limit
                    current_state = player1.AI.chessstate
                    if (current_state.get_score('W') > current_state.get_score('B')):
                        winners.append('W')
                        scores.append(current_state.get_score('W'))
                    elif (current_state.get_score('W') < current_state.get_score('B')):
                        winners.append('B')
                        scores.append(current_state.get_score('W'))
                    else:
                        winners.append('Draw')
                        scores.append(current_state.get_score('W'))
                    break
        print(winners)
        bar_width = 0.2
        for i in range(len(scores)):
            c = 'blue'
            plt.bar(i+1, height=scores[i], width=bar_width, color=c)
        plt.ylabel('Final Scores')
        plt.xlabel('Game played')
        plt.title('Problem Size ' + str(problem_size))
        plt.show()
        # Draw plots for each game, the height of the bar is the final scores that winners obtains.


def AI_match_forReal(problem_size, AI1, AI2): #Game for any two kinds of AI
    if (problem_size == 1):
        code = 'N'
    else:
        code = 'M' + str(problem_size - 2)
    if (problem_size == 1):
        mini_select = 'False'  # 8*8 game
    else:
        mini_select = 'True'  # 4*4 game
    new_game = ms1.chess_state(mini=mini_select, minitype=problem_size)
    netflags = [0, 0]
    if((AI1 == 3) or (AI2 == 3)): #We need a network
        net = chess_net(code)
        net.load_state_dict(
            tr.load('C:/Users/28529/Desktop/SU/Courses/AI/project/new code/model_' + code + '.pth'))  # Load the network
        player1 = NNbased_AI(new_game, 'W')
        print('White: NN')
        netflags[0] = 1
        if(AI1 == AI2): #Two NN-AIs
            player2 = NNbased_AI(new_game, 'B')
            print('Black: NN')
            netflags[1] = 1
        else:
            for item in [AI1, AI2]:
                if(item != 3):
                    another_select = item
                    break
            if(another_select == 1):
                player2 = ms2.treebased_AI(new_game, 'B')
                print('Black: Tree')
            else:
                player2 = ms2.random_AI(new_game, 'B')
                print('Black: Baseline')
    else:
        if(AI1 == 1):
            player1 = ms2.treebased_AI(new_game, 'W')
            print('White: Tree')
        else:
            player1 = ms2.random_AI(new_game, 'W')
            print('White: Baseline')
        if (AI2 == 1):
            player2 = ms2.treebased_AI(new_game, 'B')
            print('Black: Tree')
        else:
            player2 = ms2.random_AI(new_game, 'B')
            print('Black: Baseline')
    turn = 0
    while (turn >= 0):
        print('########## Turn: ', turn, ' ##########')
        if (turn % 2 == 0):
            if(netflags[0] == 1):
                player1.make_option(net)
            else:
                player1.make_option()
            new_state = player1.AI.chessstate
            player2.renew(new_state)  # We have to update game state for each AI once per turn
        else:
            if (netflags[1] == 1):
                player2.make_option(net)
            else:
                player2.make_option()
            new_state = player2.AI.chessstate
            player1.renew(new_state)
        key = input('Enter anything to continue: ') #Wait for inputs to go to the next turn
        new_state.show_state()
        turn += 1
        if (player1.checkmate('B') == 1):  # White wins
            print('White wins! ')
            break
        if (player2.checkmate('W') == 1):  # Black wins
            print('Black wins! ')
            break
        if (turn == 50):  # Time limit
            current_state = player1.AI.chessstate
            if (current_state.get_score('W') > current_state.get_score('B')):
                print('White wins! ')
            elif (current_state.get_score('W') < current_state.get_score('B')):
                print('Black wins! ')
            else:
                print('Draw! ')
            break


def AI_based_game(Mini, number): #Initiate a real game
    color = 'W'     #For convenience, AIs always move first
    while(1):
        print('Please select two players in the game: ')
        print(' 1.Tree Based AI, 2.Baseline AI, 3.Neural Network AI, 4.Manually')
        select1 = input('Player one: ')
        select2 = input('Player two: ')
        if((select1 in ['1', '2', '3', '4']) and (select2 in ['1', '2', '3', '4'])):
            select1 = int(select1)
            select2 = int(select2)
            if((select1 == select2) and (select1 == 4)): #Completely Manual
                ms1.game_starts(Mini,number)
                return 0
            elif((select1 != 4) and (select2 != 4)): #AI match
                AI_match_forReal(number, select1, select2)
                return 0
            else: #Human vs AI
                NN_flag = 0
                for selection in [select1, select2]:
                    if(selection != 4):
                        AI_select = selection
                        break
                if (AI_select == 1):
                    new_game = ms1.chess_state(mini=Mini, minitype=number)
                    Game_AI = ms2.treebased_AI(new_game, color)
                elif (AI_select == 2):
                    new_game = ms1.chess_state(mini=Mini, minitype=number)
                    Game_AI = ms2.random_AI(new_game, color)
                elif (AI_select == 3):
                    new_game = ms1.chess_state(mini=Mini, minitype=number)
                    Game_AI = NNbased_AI(new_game, color)
                    if (number == 1):
                        code = 'N'
                    else:
                        code = 'M' + str(number - 2)
                    net = chess_net(code)
                    net.load_state_dict(tr.load(
                        'C:/Users/28529/Desktop/SU/Courses/AI/project/new code/model_' + code + '.pth'))  # Load the network
                    NN_flag = 1
            break
        else:
            print('Invalid operation. ')

    turn = 0
    turnmod = ['B', 'W']
    moving_player = ['Black', 'White']
    AI_role = 1
    while (turn < 50):
        turn += 1
        print('####### TURN ' + str(turn) + ' #######')
        Game_AI.show_state()
        print('>>>>>>>>>' + moving_player[turn % 2] + ' Moving')
        if (moving_player[turn % 2] == moving_player[AI_role]):
            if(NN_flag == 1):
                Game_AI.make_option(net)
            else:
                Game_AI.make_option()
            valid = 1
        else:
            valid = 0
        while (valid == 0):
            print('Please give command: 1.Make a move, 2.Swap two pieces, 3.Quit')  # Select operation
            choice = input('Your Choice: ')
            if (choice == '3'):
                print('####### Game Over #######')  # Quit the game
                return 0
            elif (choice == '1'):  # Make a move
                piece = input('Type the name of piece you want to move: ')
                if (Game_AI.find_chess(piece) == 0):  # No such piece in the current list of pieces
                    print('Inoperable piece. ')
                    continue
                elif (Game_AI.find_chess(piece).side != turnmod[turn % 2]):  # The piece belongs to the opponent
                    print('Inoperable piece. ')
                    continue
                else:
                    goal = input('Type the destination where you want to move this piece: ')
                    if (len(goal) != 2):  # Incorrect format
                        print('Invalid destination. ')
                        continue
                    elif ((goal[0] not in Game_AI.AI.chessstate.cols) or (goal[1] not in Game_AI.AI.chessstate.rows)):  # No such position on chessboard
                        print('Invalid destination. ')
                        continue
                    else:
                        successful = Game_AI.AI.chessstate.one_move(piece, goal)  # Try to move the chess
                        if (successful == 0):  # Move cannot be done
                            print('Invalid operation. ')
                            continue
                        else:
                            valid = 1  # One turn is accomplished

            elif (choice == '2'):
                piece1 = input('Type the name of the first piece you want to swap: ')
                if (Game_AI.find_chess(piece1) == 0):
                    print('Inoperable piece. ')
                    continue
                elif (Game_AI.find_chess(piece1).side != turnmod[turn % 2]):
                    print('Inoperable piece. ')
                    continue
                else:
                    piece2 = input('Type the name of the second piece you want to swap: ')
                    if (Game_AI.find_chess(piece2) == 0):
                        print('Inoperable piece. ')
                        continue
                    elif (Game_AI.find_chess(piece2).side != turnmod[turn % 2]):
                        print('Inoperable piece. ')
                        continue
                    elif (piece1 == piece2):
                        print('Please enter two different pieces. ')
                        continue
                    else:
                        Game_AI.AI.chessstate.one_swap(piece1, piece2)  # Try to swap two pieces
                        valid = 1  # One turn is accomplished
            else:
                print('Invalid operation. ')
                continue
        if (Game_AI.checkmate(turnmod[(turn + 1) % 2]) == 1):  # Check if there is a checkmate
            print('>>>>>>>>>' + moving_player[turn % 2] + ' Wins! ')
            break
    print('####### Game Over #######')
    return 0


if __name__ == "__main__":
    while (1):
        size_select = input('Game Size: 1.Normal Game, 2.Mini Game one, 3.Mini Game two, 4.Mini Game three, 5.Mini Game Four ')  # Select game size
        if (size_select == '1'):  # Normal game
            AI_based_game('False', int(size_select))
            break
        elif (size_select in ['2','3','4','5']):  # Mini game
            AI_based_game('True', int(size_select))
            break
        else:
            print('Invalid operation. ')
            continue

    #AI_match_forTest()
    ## For testing AI performance: tree_based vs NN_based


    #AI_match_forData()
    #exit()
    ## For training data collection

## For model training
'''''''''''''''''''''''''''''''''
    for i in [1,2,3,4,5]:
        if(i == 1):
            problem_size = 'N'
        else:
            problem_size = 'M' + str(i-2)
        with open('C:/Users/28529/Desktop/SU/Courses/AI/project/new code/netdata_' + str(i) + '.pkl','rb') as fp:
            netdata = pickle.load(fp)
        model_training(problem_size, netdata)
'''''''''''''''''''''''''''''''''


