import cv2
import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.player_turn = True
        self.board = self.initialize_board()

    def initialize_board(self):
        return [''] * 9

    def draw_grid(self, image, contour):
        # Function to draw a 3x3 grid on the image and number the cells
        x, y, w, h = cv2.boundingRect(contour)
        cell_width = w // 3
        cell_height = h // 3

        for i in range(1, 3):
            cv2.line(image, (x, y + i * cell_height), (x + w, y + i * cell_height), (0, 255, 0), 2)

        for i in range(1, 3):
            cv2.line(image, (x + i * cell_width, y), (x + i * cell_width, y + h), (0, 255, 0), 2)

        cell_number = 1
        for i in range(3):
            for j in range(3):
                text_x = x + j * cell_width + cell_width // 2 - 10
                text_y = y + i * cell_height + cell_height // 2 + 10
                cell_number += 1

    def check_winner(self):
        # Function to check if a player has won
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != '':
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != '':
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return True
        return False

    def show_board(self):
        # Function to print game progress
        for element in [self.board[i: i + 3] for i in range(0, len(self.board), 3)]:
            print(element)

    def available_moves(self):
        return [k for k, v in enumerate(self.board) if v == '']

    def make_move(self, position, player):
        self.board[position] = player

    def play_game(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 19, 2)

            contours_grid, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = []
            for contour in contours_grid:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    if area > 1900:
                        filtered_contours.append(contour)
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        self.draw_grid(frame, contour)

            cv2.imshow("Tic Tac Toe - Detect Grid", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or len(filtered_contours) > 0:
                break

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 19, 2)

            contours_grid, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = []
            for contour in contours_grid:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    if area > 1900:
                        filtered_contours.append(contour)
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        self.draw_grid(frame, contour)

            if self.player_turn:
                self.handle_player_move(frame, x, y, w, h)
            else:
                self.handle_ai_move()

            self.draw_moves(frame, x, y, w, h)

            if self.check_winner() or all(cell != '' for cell in self.board):
                self.display_winner(frame)

            cv2.imshow("Tic Tac Toe", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or (key == ord('p') and (self.check_winner() or all(cell != '' for cell in self.board))):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def handle_player_move(self, frame, x, y, w, h):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask_red1, mask_red2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                for i in range(3):
                    for j in range(3):
                        cell_x = x + j * (w // 3)
                        cell_y = y + i * (h // 3)
                        cell_x_end = cell_x + (w // 3)
                        cell_y_end = cell_y + (h // 3)
                        if cell_x < cx < cell_x_end and cell_y < cy < cell_y_end:
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.putText(frame, "Press 'p' to confirm move", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('p'):
                                move_position = i * 3 + j
                                if self.board[move_position] == '':
                                    self.make_move(move_position, 'X')
                                    self.player_turn = False

    def handle_ai_move(self):
        move = self.determine_move()
        if self.board[move] == '':
            self.make_move(move, 'O')
            self.player_turn = True

    def determine_move(self):
        a = -2
        choices = []
        if len(self.available_moves()) == 9:
            return 4
        for move in self.available_moves():
            self.make_move(move, 'O')
            val = self.alphabeta(self, 'X', -2, 2)
            self.make_move(move, '')
            if val > a:
                a = val
                choices = [move]
            elif val == a:
                choices.append(move)
        return random.choice(choices)

    def alphabeta(self, node, player, alpha, beta):
        if node.check_winner() or all(cell != '' for cell in node.board):
            if node.check_winner():
                return -1 if player == 'O' else 1
            return 0

        for move in node.available_moves():
            node.make_move(move, player)
            val = self.alphabeta(node, 'X' if player == 'O' else 'O', alpha, beta)
            node.make_move(move, '')
            if player == 'O':
                alpha = max(alpha, val)
                if alpha >= beta:
                    return beta
            else:
                beta = min(beta, val)
                if beta <= alpha:
                    return alpha
        return alpha if player == 'O' else beta

    def draw_moves(self, frame, x, y, w, h):
        for i in range(3):
            for j in range(3):
                if self.board[i * 3 + j] == 'X':
                    cell_x = x + j * (w // 3) + (w // 3) // 2
                    cell_y = y + i * (h // 3) + (h // 3) // 2
                    cv2.line(frame, (cell_x - 15, cell_y - 15), (cell_x + 15, cell_y + 15), (0, 0, 255), 3)
                    cv2.line(frame, (cell_x - 15, cell_y + 15), (cell_x + 15, cell_y - 15), (0, 0, 255), 3)
                elif self.board[i * 3 + j] == 'O':
                    cell_x = x + j * (w // 3) + (w // 3) // 2
                    cell_y = y + i * (h // 3) + (h // 3) // 2
                    cv2.circle(frame, (cell_x, cell_y), 30, (255, 0, 0), 3)

    def display_winner(self, frame):
        winner = self.check_winner()
        if winner == 'X':
            cv2.putText(frame, "Player wins!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif winner == 'O':
            cv2.putText(frame, "AI wins!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "It's a draw!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to exit", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

if __name__ == "__main__":
    game = TicTacToe()
    game.play_game()
