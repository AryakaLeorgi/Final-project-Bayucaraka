#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm> // Added for std::find

class Tic {
private:
    std::vector<std::vector<int>> winning_combos = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
        {0, 4, 8}, {2, 4, 6}
    };
    std::vector<int> squares;

public:
    Tic() : squares(9, 0) {}

    void show() {
        for (int i = 0; i < 9; i += 3) {
            for (int j = 0; j < 3; ++j) {
                if (squares[i + j] == 1) std::cout << "X | ";
                else if (squares[i + j] == 2) std::cout << "O | ";
                else std::cout << "- | ";
            }
            std::cout << std::endl << "---------" << std::endl;
        }
    }

    std::vector<int> available_moves() {
        std::vector<int> moves;
        for (int i = 0; i < 9; ++i) {
            if (squares[i] == 0) {
                moves.push_back(i);
            }
        }
        return moves;
    }

    bool complete() {
        if (std::find(squares.begin(), squares.end(), 0) == squares.end() || winner() != 0) {
            return true;
        }
        return false;
    }

    bool X_won() {
        return winner() == 1;
    }

    bool O_won() {
        return winner() == 2;
    }

    bool tied() {
        return complete() && winner() == 0;
    }

    int winner() {
        for (int player : {1, 2}) {
            for (auto combo : winning_combos) {
                bool win = true;
                for (int pos : combo) {
                    if (squares[pos] != player) {
                        win = false;
                        break;
                    }
                }
                if (win) {
                    return player;
                }
            }
        }
        return 0;
    }

    void make_move(int position, int player) {
        squares[position] = player;
    }

    int alphabeta(Tic node, int player, int alpha, int beta) {
        if (node.complete()) {
            if (node.X_won()) return -1;
            else if (node.tied()) return 0;
            else if (node.O_won()) return 1;
        }
        for (int move : node.available_moves()) {
            node.make_move(move, player);
            int val = alphabeta(node, get_enemy(player), alpha, beta);
            node.make_move(move, 0);
            if (player == 2) {
                if (val > alpha) alpha = val;
                if (alpha >= beta) return beta;
            } else {
                if (val < beta) beta = val;
                if (beta <= alpha) return alpha;
            }
        }
        return player == 2 ? alpha : beta;
    }

    int get_enemy(int player) {
        return player == 1 ? 2 : 1;
    }
};

int determine(Tic board, int player) {
    int a = -2;
    std::vector<int> choices;
    if (board.available_moves().size() == 9) return 4;
    for (int move : board.available_moves()) {
        board.make_move(move, player);
        int val = board.alphabeta(board, board.get_enemy(player), -2, 2);
        board.make_move(move, 0);
        if (val > a) {
            a = val;
            choices = {move};
        } else if (val == a) {
            choices.push_back(move);
        }
    }
    return choices[rand() % choices.size()];
}

class TicTacToePublisher : public rclcpp::Node {
public:
    TicTacToePublisher() : Node("tic_tac_toe_publisher") {
        publisher_ = this->create_publisher<std_msgs::msg::String>("tic_tac_toe", 10);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(500), std::bind(&TicTacToePublisher::publish_state, this));
    }

private:
    void publish_state() {
        auto message = std_msgs::msg::String();
        message.data = "Game State"; // Replace this with the actual game state string
        publisher_->publish(message);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TicTacToePublisher>());
    rclcpp::shutdown();
    return 0;
}
