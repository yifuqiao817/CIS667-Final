#### CIS 667
#### Yifu Qiao - yqiao08
#### Project Milestone 1
import numpy as np
import random as rd

White_chess = ['WK ','WQ ','WP1','WP2','WP3','WP4','WP5','WP6','WP7','WP8','WR1','WR2','WB1','WB2','WN1','WN2']
Black_chess = ['BK ','BQ ','BP1','BP2','BP3','BP4','BP5','BP6','BP7','BP8','BR1','BR2','BB1','BB2','BN1','BN2']
#Pieces for normal game

White_chess_44 = [['WK ','WQ ','WR1','WR2'],['WK ','WQ ','WN1','WN2'],['WK ','WQ ','WB1','WB2'],['WK ','WQ ','WP1','WP2']]
Black_chess_44 = [['BK ','BQ ','BR1','BR2'],['BK ','BQ ','BN1','BN2'],['BK ','BQ ','BB1','BB2'],['BK ','BQ ','BP1','BP2']]
#Pieces for mini game

columns = ['A','B','C','D','E','F','G','H'] #Mark the position for each cell of the chessboard
rows = [str(num+1) for num in range(8)]
class chess_state(object):
    def __init__(self,mini='False',minitype=0):
        self.mini = mini
        self.minitype = minitype-2
        if(mini == 'True'): #Create a mini game
            self.size = 'M'
            self.cols = columns[:4]
            self.rows = rows[:4]
            self.shape = (4,4)
            self.chessboard = np.empty((4,4),dtype='S3')
            for row in range(4):
                for col in range(4):
                    self.chessboard[row, col] = '   '
            self.pieces = []
            for piece in White_chess_44[minitype-2] + Black_chess_44[minitype-2]:
                self.pieces.append(chess(piece,self.size))
            for newpiece in self.pieces:
                r, c = newpiece.num_posit()
                self.chessboard[r, c] = newpiece.str_name()
        else:
            self.size = 'N' #Create a normal game
            self.cols = columns
            self.rows = rows
            self.shape = (8,8)
            self.chessboard = np.empty((8, 8), dtype='S3') #Where we store where the pieces are
            for row in range(8):
                for col in range(8):
                    self.chessboard[row, col] = '   ' #Initialization
            self.pieces = []
            for piece in White_chess + Black_chess:
                self.pieces.append(chess(piece,self.size)) #Create every piece
            for newpiece in self.pieces:
                r, c = newpiece.num_posit()
                self.chessboard[r, c] = newpiece.str_name() #Put pieces into place


    def show_state(self): #Display the chessboard
        if(self.size == 'N'):
            print('  | A | B | C | D | E | F | G | H |  ')
            for row in range(8):
                print(' ' + str(8 - row), end='|')
                for col in range(8):
                    print(str(self.chessboard[row, col], encoding='utf-8'), end='|')
                print(str(8 - row) + ' ')
            print('  | A | B | C | D | E | F | G | H |  ')
        else:
            print('  | A | B | C | D |  ')
            for row in range(4):
                print(' ' + str(4 - row), end='|')
                for col in range(4):
                    print(str(self.chessboard[row, col], encoding='utf-8'), end='|')
                print(str(4 - row) + ' ')
            print('  | A | B | C | D |  ')


    def get_score(self, side): #side: 'W', 'B', calculate score by sum up the scores for each piece
        all_score = 0
        temp_measure = 1
        for piece in self.pieces:
            if(piece.side == side):
                temp_measure = 1
            elif(piece.side != side):
                temp_measure = -1
            all_score += temp_measure * piece.score
        return  all_score


    def one_move(self,piece,goal): #Example: BN2 C6
        success_flag = 0
        whethertoeat = 0
        if (piece in ['WK', 'WQ', 'BQ', 'BK']): #In case player did not type space
            piece = piece + ' '
        for currentp in self.pieces:
            if(piece == currentp.str_name()): #Find the piece to move
                #print('&&&&&& Piece found')
                old_r, old_c = currentp.num_posit()
                est_r, est_c = posit_trans(goal)
                if(self.size == 'M'): #For minigame the index is different
                    est_r = 4 - int(goal[1])
                #print('&&&&&& Coordinate extracted')
                if (str(self.chessboard[est_r, est_c],encoding='utf-8') != '   '): #Another piece on the goal
                    #print('&&&&&& Target Occpupied')
                    if(self.find_chess(str(self.chessboard[est_r, est_c],encoding='utf-8')).side == currentp.side):
                        #Cannot take the piece with the same color
                        return 0
                    else:
                        #Piece in the goal is opponent
                        whethertoeat = 1
                operable, trace = currentp.chess_move(goal,whethertoeat)
                #operable = whether the move fit the moving rule of the piece
                #trace = the cells on the moving direction from current position to the target
                if(operable == 1):
                    for (row, col) in trace: #check the cells on the trace are all empty
                        if (str(self.chessboard[row, col], encoding='utf-8') != '   '):
                            return 0
                    new_r, new_c =currentp.num_posit() #new index
                    self.chessboard[old_r,old_c] = '   ' #current position is empty
                    if(str(self.chessboard[new_r,new_c],encoding='utf-8') != '   '):
                        self.del_chess(str(self.chessboard[new_r,new_c],encoding='utf-8'))
                        #Take the opponent's piece if there is one
                    self.chessboard[new_r,new_c] = currentp.str_name()
                    #Update the chessboard
                    success_flag = 1 #Move successfully
                break
        return success_flag

    def fake_move(self,currentp,goal): #To check the possible steps when we check if there is a checkmate
        success_flag, trace = currentp.fake_move(goal)
        if (success_flag == 1):
            for (row, col) in trace:
                if (str(self.chessboard[row, col], encoding='utf-8') != '   '):
                    return 0
        return success_flag

    def fake_move_for_AI(self,currentp,goal):
        success_flag, trace = currentp.fake_move(goal)
        old_r, old_c = currentp.num_posit()
        est_r, est_c = posit_trans(goal)
        if (self.size == 'M'):  # For minigame the index is different
            est_r = 4 - int(goal[1])
        if(currentp.type == 'Pawn'):
            if(currentp.name[0] == 'W'):
                if((abs(est_c - old_c) == 1) and (est_r - old_r == -1)):
                    if (str(self.chessboard[est_r, est_c], encoding='utf-8') != '   '):  # Another piece on the goal
                        if (self.find_chess(str(self.chessboard[est_r, est_c], encoding='utf-8')).side != currentp.side):
                            return 1
                elif((abs(est_c - old_c) == 0) and (est_r - old_r == -1)):
                    if (str(self.chessboard[est_r, est_c], encoding='utf-8') != '   '):
                        return 0
                    else:
                        return 1
                elif((currentp.moved == 0) and (abs(est_c - old_c) == 0) and (est_r - old_r == -2)):
                    if (str(self.chessboard[est_r, est_c], encoding='utf-8') != '   '):
                        return 0
                    elif(str(self.chessboard[est_r + 1, est_c], encoding='utf-8') != '   '):
                        return 0
                    else:
                        return 1
            elif (currentp.name[0] == 'B'):
                if ((abs(est_c - old_c) == 1) and (est_r - old_r == 1)):
                    if (str(self.chessboard[est_r, est_c], encoding='utf-8') != '   '):  # Another piece on the goal
                        if (self.find_chess(
                                str(self.chessboard[est_r, est_c], encoding='utf-8')).side != currentp.side):
                            return 1
                elif ((abs(est_c - old_c) == 0) and (est_r - old_r == 1)):
                    if (str(self.chessboard[est_r, est_c], encoding='utf-8') != '   '):
                        return 0
                    else:
                        return 1
                elif((currentp.moved == 0) and (abs(est_c - old_c) == 0) and (est_r - old_r == 2)):
                    if (str(self.chessboard[est_r, est_c], encoding='utf-8') != '   '):
                        return 0
                    elif(str(self.chessboard[est_r - 1, est_c], encoding='utf-8') != '   '):
                        return 0
                    else:
                        return 1
            return 0


        if (str(self.chessboard[est_r, est_c], encoding='utf-8') != '   '):  # Another piece on the goal
            if (self.find_chess(str(self.chessboard[est_r, est_c], encoding='utf-8')).side == currentp.side):
                return 0
        if (success_flag == 1):
            for (row, col) in trace:
                if (str(self.chessboard[row, col], encoding='utf-8') != '   '):
                    return 0
        return success_flag

    def one_swap(self,piece1,piece2): #Swap two pieces
        p1 = self.find_chess(piece1)
        p2 = self.find_chess(piece2)
        r1, c1 = p1.num_posit()
        r2, c2 = p2.num_posit()
        pos1 = p1.curr_pos
        pos2 = p2.curr_pos
        self.chessboard[r1,c1] = p2.str_name() #Update chessboard state
        self.chessboard[r2,c2] = p1.str_name()
        p1.change_posit(pos2)
        p2.change_posit(pos1)
        self.del_chess(piece1) #Delete the old pieces
        self.del_chess(piece2)
        self.pieces.append(p1) #Update list of pieces
        self.pieces.append(p2)
        return 0

    def find_chess(self, called): #Given the name of a piece, find the object in the list of pieces
        if(called in ['WK','WQ','BQ','BK']): #In case that player did not type in space
            called = called+' '
        for survived in self.pieces:
            if(called == survived.str_name()):
                return survived
        return 0

    def del_chess(self,slayed): #Pop out the piece that has been taken (or swapped)
        newlist = []
        for survived in self.pieces:
            if(slayed != survived.str_name()):
                newlist.append(survived)
        self.pieces = newlist
        return 0

    def checkmate(self,whosturn): #Check if there is a checkmate
        check = 0
        if(whosturn == 'B'): #Check who's turn, to determine the winner
            king = self.find_chess('BK ')
        else:
            king = self.find_chess('WK ')
        if(king == 0):
            return 1 #King is been taken
        kr, kc = king.num_posit()
        escape_count = 0
        check_count = 0
        for newr, newc in [(kr,kc),(kr,kc+1),(kr,kc-1),(kr+1,kc),(kr+1,kc+1),(kr+1,kc-1),(kr-1,kc),(kr-1,kc+1),(kr-1,kc-1)]:
            #All the possibile moves for king
            if((newr in range(self.shape[0])) and (newc in range(self.shape[1]))): #Still on the chessboard
                if ((str(self.chessboard[newr, newc], encoding='utf-8') == '   ') or ((newr,newc) == (kr,kc))): #Move or stay
                    king_pos = posit_tostr(newr,newc)
                    if(self.size == 'M'): #Position for minigame
                        king_pos = columns[newc] + str(4 - newr)
                    escape_count += 1
                    for survived in self.pieces:
                        if(survived.side != king.side):
                            if(self.fake_move(survived,king_pos) == 1):
                                check_count += 1
                                break
        if((escape_count>0) and (check_count == escape_count)): #No way for king to escape
            check = 1 #Checkmate
        return check

class chess(object): #Class definition for each piece
    def __init__(self, name, size):
        self.type = '' #King, Queen, Pawn, Rook, Bishop, Knight
        self.init_pos = '' #Initial position on the chessboard, like 'A1'
        self.name = name #String name, like 'WR1' for white rook piece No.1.
        self.size = size #To check whether it is on the minigame.
        self.moved = 0 #Whether it has been moved at least once.
        if(name in ['WK ','BK ']):
            self.type = 'King'
            self.score = 900
            if(name.startswith('W')):
                self.init_pos = 'D1'
                if(size == 'M'):
                    self.init_pos = 'B1'
            else:
                self.init_pos = 'D8'
                if(size == 'M'):
                    self.init_pos = 'B4'

        elif(name in ['WQ ','BQ ']):
            self.type = 'Queen'
            self.score = 90
            if (name.startswith('W')):
                self.init_pos = 'E1'
                if(size == 'M'):
                    self.init_pos = 'C1'
            else:
                self.init_pos = 'E8'
                if(size == 'M'):
                    self.init_pos = 'C4'

        elif ((name.startswith('BP')) or (name.startswith('WP'))):
            self.type = 'Pawn'
            self.score = 10
            if (name.startswith('W')):
                self.init_pos = columns[int(name[2]) - 1]+'2'
                if(size == 'M'):
                    if(name == 'WP1'):
                        self.init_pos = 'A1'
                    elif(name == 'WP2'):
                        self.init_pos = 'D1'
            else:
                self.init_pos = columns[8 - int(name[2])]+'7'
                if (size == 'M'):
                    if (name == 'BP1'):
                        self.init_pos = 'D4'
                    elif (name == 'BP2'):
                        self.init_pos = 'A4'

        elif ((name.startswith('BR')) or (name.startswith('WR'))):
            self.type = 'Rook'
            self.score = 50
            if (name == 'WR1'):
                self.init_pos = 'A1'
            elif (name == 'WR2'):
                self.init_pos = 'H1'
                if(size == 'M'):
                    self.init_pos = 'D1'
            elif (name == 'BR1'):
                self.init_pos = 'H8'
                if (size == 'M'):
                    self.init_pos = 'D4'
            elif (name == 'BR2'):
                self.init_pos = 'A8'
                if (size == 'M'):
                    self.init_pos = 'A4'

        elif ((name.startswith('BB')) or (name.startswith('WB'))):
            self.type = 'Bishop'
            self.score = 30
            if (name == 'WB1'):
                self.init_pos = 'C1'
                if(size == 'M'):
                    self.init_pos = 'A1'
            elif (name == 'WB2'):
                self.init_pos = 'F1'
                if (size == 'M'):
                    self.init_pos = 'D1'
            elif (name == 'BB1'):
                self.init_pos = 'F8'
                if (size == 'M'):
                    self.init_pos = 'D4'
            elif (name == 'BB2'):
                self.init_pos = 'C8'
                if (size == 'M'):
                    self.init_pos = 'A4'

        elif ((name.startswith('BN')) or (name.startswith('WN'))):
            self.type = 'Knight'
            self.score = 30
            if (name == 'WN1'):
                self.init_pos = 'B1'
                if (size == 'M'):
                    self.init_pos = 'A1'
            elif (name == 'WN2'):
                self.init_pos = 'G1'
                if (size == 'M'):
                    self.init_pos = 'D1'
            elif (name == 'BN1'):
                self.init_pos = 'G8'
                if (size == 'M'):
                    self.init_pos = 'D4'
            elif (name == 'BN2'):
                self.init_pos = 'B8'
                if (size == 'M'):
                    self.init_pos = 'A4'

        self.curr_pos = self.init_pos
        self.side = name[0]

    def num_posit(self): #Change the position name like 'A1' to numeric chessboard index.
        if(self.size == 'M'): #For minigame
            col = columns.index(self.curr_pos[0])
            row = 4 - int(self.curr_pos[1])
            return (row,col)
        return posit_trans(self.curr_pos) #For normal game

    def str_name(self):
        return self.name

    def change_posit(self,called): #Change the position of the piece
        self.curr_pos = called
        self.moved = 1
        return 0

    def chess_move(self, dest, eat): #Moving rules for each piece
        cando = 0 #Whether it is operable
        trace = [] #The cells on its way to the destination
        goalr, goalc = posit_trans(dest)
        if(self.size == 'M'): #Goals for minigame
            goalc = columns.index(dest[0])
            goalr = 4 - int(dest[1])
        fromr, fromc = self.num_posit()
        if(self.type == 'Pawn'): #Pawn
            if(eat == 1): #To take opponent's piece, it moves diagonally.
                if(self.side == 'W'):
                    if((goalr - fromr) == -1):
                        if(abs(fromc - goalc) == 1):
                            cando = 1
                if (self.side == 'B'):
                    if ((goalr - fromr) == 1):
                        if (abs(fromc - goalc) == 1):
                            cando = 1
            else:
                if(self.moved == 0): #Pawn's first move can have two steps.
                    if (self.side == 'W'):
                        if ((goalr - fromr) in [-1,-2]):
                            if (fromc - goalc == 0):
                                cando = 1
                                trace.append((fromr - 1, fromc))
                                if(goalr - fromr == -2):
                                    trace.append((fromr - 2, fromc))

                    if (self.side == 'B'):
                        if ((goalr - fromr) in [1,2]):
                            if (fromc - goalc == 0):
                                cando = 1
                                trace.append((fromr + 1, fromc))
                                if (goalr - fromr == 2):
                                    trace.append((fromr + 2, fromc))
                else:
                    if (self.side == 'W'):
                        if ((goalr - fromr) == -1):
                            if (fromc - goalc == 0):
                                cando = 1
                    if (self.side == 'B'):
                        if ((goalr - fromr) == 1):
                            if (fromc - goalc == 0):
                                cando = 1

        elif(self.type == 'King'): #King
            if ((abs(goalr - fromr) <= 1) and (abs(fromc - goalc) <= 1)):
                cando = 1

        elif (self.type == 'Queen'): #Queen
            rmove = abs(goalr - fromr)
            cmove = abs(goalc - fromc)
            if ((rmove + cmove > 0) and (rmove * cmove == 0)): #Like Rook's move
                cando = 1
                if(cmove == 0):
                    if(goalr - 1 > fromr):
                        for i in range(fromr+1,goalr):
                            trace.append((i,fromc))
                    elif(fromr - 1 > goalr):
                        for i in range(goalr+1,fromr):
                            trace.append((i,fromc))
                if (rmove == 0):
                    if (goalc - 1 > fromc):
                        for j in range(fromc + 1, goalc):
                            trace.append((fromr,j))
                    elif (fromc - 1 > goalc):
                        for j in range(goalc + 1, fromc):
                            trace.append((fromr,j))

            if (rmove == cmove): #Like Bishop's move
                cando = 1
                rdir = (goalr - fromr) / rmove
                cdir = (goalc - fromc) / cmove
                r = fromr
                c = fromc
                while(1):
                    r += rdir
                    c += cdir
                    if((r == goalr) or (c == goalc)):
                        break
                    trace.append((int(r),int(c)))

        elif (self.type == 'Knight'): #Knight
            rmove = abs(goalr - fromr)
            cmove = abs(goalc - fromc)
            if (((rmove == 2) and (cmove == 1)) or ((rmove == 1) and (cmove == 2))):
                cando = 1

        elif (self.type == 'Rook'): #Rook
            rmove = abs(goalr - fromr)
            cmove = abs(goalc - fromc)
            if ((rmove + cmove > 0) and (rmove * cmove == 0)):
                cando = 1
                if (cmove == 0):
                    if (goalr - 1 > fromr):
                        for i in range(fromr + 1, goalr):
                            trace.append((i, fromc))
                    elif (fromr - 1 > goalr):
                        for i in range(goalr + 1, fromr):
                            trace.append((i, fromc))
                if (rmove == 0):
                    if (goalc - 1 > fromc):
                        for j in range(fromc + 1, goalc):
                            trace.append((fromr, j))
                    elif (fromc - 1 > goalc):
                        for j in range(goalc + 1, fromc):
                            trace.append((fromr, j))

        elif (self.type == 'Bishop'): #Bishop
            rmove = abs(goalr - fromr)
            cmove = abs(goalc - fromc)
            if (rmove == cmove):
                cando = 1
                rdir = (goalr - fromr) / rmove
                cdir = (goalc - fromc) / cmove
                r = fromr
                c = fromc
                while (1):
                    r += rdir
                    c += cdir
                    if ((r == goalr) or (c == goalc)):
                        break
                    trace.append((int(r), int(c)))

        if(cando == 1): #It is operable
            self.curr_pos = dest
            self.moved = 1
        return cando,trace

    def fake_move(self, dest): #The only difference from chess_move is that we do not change the state of the piece.
        #Only to see whether the move fits the moving rule of the piece.
        cando = 0
        trace = []
        goalr, goalc = posit_trans(dest)
        fromr, fromc = self.num_posit()
        if (self.size == 'M'):
            goalc = columns.index(dest[0])
            goalr = 4 - int(dest[1])
        if(self.type == 'Pawn'):
            if (self.side == 'W'):
                if ((goalr - fromr) == -1):
                    if (abs(fromc - goalc) == 1):
                        cando = 1
            if (self.side == 'B'):
                if ((goalr - fromr) == 1):
                    if (abs(fromc - goalc) == 1):
                        cando = 1

        elif(self.type == 'King'):
            if ((abs(goalr - fromr) <= 1) and (abs(fromc - goalc) <= 1)):
                cando = 1

        elif (self.type == 'Queen'):
            rmove = abs(goalr - fromr)
            cmove = abs(goalc - fromc)
            if ((rmove + cmove > 0) and (rmove * cmove == 0)): #Rook move
                cando = 1
                if(cmove == 0):
                    if(goalr - 1 > fromr):
                        for i in range(fromr+1,goalr):
                            trace.append((i,fromc))
                    elif(fromr - 1 > goalr):
                        for i in range(goalr+1,fromr):
                            trace.append((i,fromc))
                if (rmove == 0):
                    if (goalc - 1 > fromc):
                        for j in range(fromc + 1, goalc):
                            trace.append((fromr,j))
                    elif (fromc - 1 > goalc):
                        for j in range(goalc + 1, fromc):
                            trace.append((fromr,j))
            if (rmove == cmove): #Bishop move
                cando = 1
                rdir = (goalr - fromr) / rmove
                cdir = (goalc - fromc) / cmove
                r = fromr
                c = fromc
                while(1):
                    r += rdir
                    c += cdir
                    if((r == goalr) or (c == goalc)):
                        break
                    trace.append((int(r),int(c)))

        elif (self.type == 'Knight'):
            rmove = abs(goalr - fromr)
            cmove = abs(goalc - fromc)
            if (((rmove == 2) and (cmove == 1)) or ((rmove == 1) and (cmove == 2))):
                cando = 1

        elif (self.type == 'Rook'):
            rmove = abs(goalr - fromr)
            cmove = abs(goalc - fromc)
            if ((rmove + cmove > 0) and (rmove * cmove == 0)):
                cando = 1
                if (cmove == 0):
                    if (goalr - 1 > fromr):
                        for i in range(fromr + 1, goalr):
                            trace.append((i, fromc))
                    elif (fromr - 1 > goalr):
                        for i in range(goalr + 1, fromr):
                            trace.append((i, fromc))
                if (rmove == 0):
                    if (goalc - 1 > fromc):
                        for j in range(fromc + 1, goalc):
                            trace.append((fromr, j))
                    elif (fromc - 1 > goalc):
                        for j in range(goalc + 1, fromc):
                            trace.append((fromr, j))

        elif (self.type == 'Bishop'):
            rmove = abs(goalr - fromr)
            cmove = abs(goalc - fromc)
            if (rmove == cmove):
                cando = 1
                rdir = (goalr - fromr) / rmove
                cdir = (goalc - fromc) / cmove
                r = fromr
                c = fromc
                while (1):
                    r += rdir
                    c += cdir
                    if ((r == goalr) or (c == goalc)):
                        break
                    trace.append((int(r), int(c)))
        return cando,trace

def posit_trans(cell): #String position to numeric position
    col = columns.index(cell[0])
    row = 8 - int(cell[1])
    return (row, col)

def posit_tostr(r,c): #Numeric position to string position
    pos0 = columns[c]
    pos1 = str(8-r)
    return pos0+pos1

def game_starts(mini,minitype):

    game = chess_state(mini=mini,minitype=minitype) #Start the game
    turn = 0
    turnmod = ['B','W']
    moving_player = ['Black', 'White']
    # To determine which player's turn it is. White goes first.

    while(turn < 50):
        turn += 1
        print('####### TURN ' + str(turn) + ' #######')
        game.show_state()
        print('>>>>>>>>>'+moving_player[turn % 2]+' Moving')
        valid = 0
        while(valid == 0):
            print('Please give command: 1.Make a move, 2.Swap two pieces, 3.Quit') #Select operation
            choice = input('Your Choice: ')
            if (choice == '3'):
                print('####### Game Over #######') #Quit the game
                return 0
            elif (choice == '1'): #Make a move
                piece = input('Type the name of piece you want to move: ')
                if (game.find_chess(piece) == 0): #No such piece in the current list of pieces
                    print('Inoperable piece. ')
                    continue
                elif (game.find_chess(piece).side != turnmod[turn % 2]): #The piece belongs to the opponent
                    print('Inoperable piece. ')
                    continue
                else:
                    goal = input('Type the destination where you want to move this piece: ')
                    if (len(goal) != 2): #Incorrect format
                        print('Invalid destination. ')
                        continue
                    elif ((goal[0] not in game.cols) or (goal[1] not in game.rows)): #No such position on chessboard
                        print('Invalid destination. ')
                        continue
                    else:
                        successful = game.one_move(piece, goal) #Try to move the chess
                        if (successful == 0): #Move cannot be done
                            print('Invalid operation. ')
                            continue
                        else:
                            valid = 1 #One turn is accomplished

            elif (choice == '2'):
                piece1 = input('Type the name of the first piece you want to swap: ')
                if (game.find_chess(piece1) == 0):
                    print('Inoperable piece. ')
                    continue
                elif (game.find_chess(piece1).side != turnmod[turn % 2]):
                    print('Inoperable piece. ')
                    continue
                else:
                    piece2 = input('Type the name of the second piece you want to swap: ')
                    if (game.find_chess(piece2) == 0):
                        print('Inoperable piece. ')
                        continue
                    elif (game.find_chess(piece2).side != turnmod[turn % 2]):
                        print('Inoperable piece. ')
                        continue
                    elif (piece1 == piece2):
                        print('Please enter two different pieces. ')
                        continue
                    else:
                        game.one_swap(piece1, piece2) #Try to swap two pieces
                        valid = 1 #One turn is accomplished
            else:
                print('Invalid operation. ')
                continue
        if(game.checkmate(turnmod[(turn + 1) % 2]) == 1): #Check if there is a checkmate
            print('>>>>>>>>>' + moving_player[turn % 2] + ' Wins! ')
            break
    print('####### Game Over #######')
    return 0



if __name__ == "__main__":
    while(1):
        size_select = input('Game Size: 1.Normal Game, 2.Mini Game: ') #Select game size
        if(size_select == '1'): #Normal game
            game_starts('False',0)
            break
        elif(size_select == '2'): #Mini game
            game_starts('True',2)
            break
        else:
            print('Invalid operation. ')
            continue