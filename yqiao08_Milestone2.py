#### CIS 667
#### Yifu Qiao - yqiao08
#### Project Milestone 1

import yqiao08_Milestone1 as ms1 #Import the chess states
import random as rd
import copy
import matplotlib.pyplot as plt
import numpy as np


class AI_choices(object): #Represent a node in the tree
    def __init__(self,chess,color):
        self.chessstate = chess #The state of the chessboard
        self.side = color #The side which AI plays
        self.score = 0 #The score of the node in Minimax algorithm
        self.children = [] #Children of the node in the tree
        self.move = (0,0,0) #How did the parent move to the current node
                            #(Piece, Target, 0) for moving or (Piece1, Piece2, 1) for swapping
    def copy(self):
        return AI_choices(chess=self.chessstate, color=self.side)

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

class treebased_AI(object): #Tree based AI
    def __init__(self, chess, color):
        self.AI = AI_choices(chess,color)

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
                        new_AI = AI_choices(chess=current_state, color=opposite_side(current_side))
                        new_AI.move = op
                    elif(op[2] == 1): #Swapping option
                        current_state.one_swap(op[0], op[1])
                        new_AI = AI_choices(chess=current_state, color=opposite_side(current_side))
                        new_AI.move = op
                    one_node.children.append(copy.copy(new_AI))
                    new_nodes.append(copy.copy(new_AI)) #Not deep copy for we want to get access to the children
            nodes.append(new_nodes)
            nodes_count += len(new_nodes)
            i += 1
        print('There are ' + str(nodes_count) + ' nodes for the tree to process')
        master_side = temp_AI.side
        final_score = nodes[0][0].minimax(master_side)
        candidates = []
        for child in nodes[0][0].children: #Choose the next step
            if(child.score == final_score):
                candidates.append(child)
        if (len(candidates) == 1):
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



class random_AI(object): #Take actions randomly
    def __init__(self, chess, color):
        self.AI = AI_choices(chess,color)

    def renew(self, new_state):
        self.AI.newstate(new_state)

    def make_option(self):
        op_select = rd.randint(0,1) #Equal chance to either move or swap
        possible_options = self.AI.possible_options()
        rd.shuffle(possible_options)
        for selection in possible_options:
            if(selection[2] == op_select):
                one_selection = selection
        try: #In case there is no selection for moving or swapping
            print(one_selection)
        except:
            one_selection = possible_options[0]
        if(one_selection[2] == 0):
            self.AI.chessstate.one_move(one_selection[0],one_selection[1])
        elif(one_selection[2] == 1):
            self.AI.chessstate.one_swap(one_selection[0],one_selection[1])

    def show_state(self):
        self.AI.chessstate.show_state()

    def checkmate(self,whosturn):
        kingname = whosturn + 'K'
        if (self.find_chess(kingname) == 0):
            return 1
        else:
            return 0

    def find_chess(self,target):
        return self.AI.chessstate.find_chess(target)


def AI_based_game(Mini, number, side): #Human vs AI
    color = ''
    if(side == '1'):
        color = 'B' #Because you have entered your side, then we pass AI the opposite side.
    elif(side == '2'):
        color = 'W'
    while(1):
        AI_select = input('AI: 1.Tree Based, 2.Random Choice') #Choose the type of AI to fight
        if (AI_select == '1'):
            new_game = ms1.chess_state(mini=Mini, minitype=number)
            Game_AI = treebased_AI(new_game, color)
            break
        elif (AI_select == '2'):
            new_game = ms1.chess_state(mini=Mini,minitype=number)
            Game_AI = random_AI(new_game,color)
            break
        else:
            print('Invalid operation. ')

    turn = 0
    turnmod = ['B', 'W']
    moving_player = ['Black', 'White']
    AI_role = int(side) - 1
    #Player_role = abs(int(side) - 3) - 1
    while (turn < 50):
        turn += 1
        print('####### TURN ' + str(turn) + ' #######')
        Game_AI.show_state()
        print('>>>>>>>>>' + moving_player[turn % 2] + ' Moving')
        if (moving_player[turn % 2] == moving_player[AI_role]):
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


def AI_match(): #Tree based AI vs Baseline AI
    White_count = [0,0,0,0,0]
    Black_count = [0,0,0,0,0]
    Draw_count = [0,0,0,0,0]
    for problem_size in [1,2,3,4,5]:
        winners = []
        for game_num in range(100):
            if (problem_size == 1):
                mini_select = 'False' #8*8 game
            else:
                mini_select = 'True' #4*4 game
            new_game = ms1.chess_state(mini=mini_select, minitype=problem_size)
            player1 = random_AI(new_game, 'W')
            player2 = treebased_AI(new_game, 'B')
            turn = 0
            while (turn >= 0):
                print(problem_size,turn,winners)
                if (turn % 2 == 0):
                    player1.make_option()
                    new_state = player1.AI.chessstate
                    player2.renew(new_state) #We have to update game state for each AI once per turn
                else:
                    player2.make_option()
                    new_state = player2.AI.chessstate
                    player1.renew(new_state)
                new_state.show_state()
                turn += 1
                if(player1.checkmate('B') == 1): #White wins
                    winners.append('W')
                    White_count[problem_size - 1] += 1
                    break
                if(player2.checkmate('W') == 1): #Black wins
                    winners.append('B')
                    Black_count[problem_size - 1] += 1
                    break
                if(turn == 50): #Time limit
                    current_state = player1.AI.chessstate
                    if(current_state.get_score('W') > current_state.get_score('B')):
                        winners.append('W')
                        White_count[problem_size - 1] += 1
                    elif(current_state.get_score('W') < current_state.get_score('B')):
                        winners.append('B')
                        Black_count[problem_size - 1] += 1
                    else:
                        winners.append('Draw')
                        Draw_count[problem_size - 1] += 1
                    break
        print(winners)
    label_list = ['Problem 1', 'Problem 2', 'Problem 3', 'Problem 4', 'Problem 5'] #Draw a figure to show results
    x = range(5)
    bar_width = 0.2
    index_1 = np.arange(len(White_count))
    index_2 = index_1 + bar_width
    index_3 = index_2 + bar_width
    plt.bar(index_1, height=White_count, width=bar_width, color='red', label='Baseline Wins')
    plt.bar(index_2, height=Black_count, width=bar_width, color='blue', label='Tree Wins')
    plt.bar(index_3, height=Draw_count, width=bar_width, color='green', label='Draw')
    plt.ylim(0, 100)
    plt.ylabel('Winning numbers')
    plt.xlabel('Problem sizes')
    plt.legend()
    plt.xticks(index_2, label_list)
    plt.show()



if __name__ == "__main__":

    #AI_match() #Use this to make two ai battle again.
    #exit()
    while (1):
        side_select = input('Which side you want to play: 1.White (moves first), 2.Black')
        if(side_select in ['1', '2']):
            break
        else:
            print('Invalid operation. ')
            continue

    while (1):
        size_select = input('Game Size: 1.Normal Game, 2.Mini Game one, 3.Mini Game two, 4.Mini Game three, 5.Mini Game Four ')  # Select game size
        if (size_select == '1'):  # Normal game
            AI_based_game('False', int(size_select),side_select)
            break
        elif (size_select in ['2','3','4','5']):  # Mini game
            AI_based_game('True', int(size_select),side_select)
            break
        else:
            print('Invalid operation. ')
            continue



