import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Literal
import random


class Maze:
    """
    Class for creating and solving mazes.
    """

    def __init__(self):
        pass

    def load_maze(self, file_name: str) -> np.ndarray:
        """
        This method loads a maze from a file stored in nd.array.
        """
        maze = np.loadtxt(file_name, delimiter=",", dtype=bool)
        return maze

    def generate_temple(self, size: int, mode: Literal['empty', 'slalom',  'ess', 'essthin'] = "empty") -> np.ndarray:
        """
        This method generates maze templete with given size and mode.
        There are three modes: empty, slalom, ess.
        """
        maze = np.zeros((size, size), dtype=bool)
        if mode == "empty":
            pass
        elif mode == "slalom":
            for i in range(1, size - 1):
                if i % 10 == 0:
                    maze[i, :size-1] = 1
                elif i % 5 == 0:
                    maze[i, 1:size] = 1
        elif mode == "ess":
            fifth = size//5
            for i in range(fifth, 2*fifth):
                maze[i, :fifth*4] = 1
            for i in range(3*fifth, 4*fifth):
                maze[i, fifth:size] = 1
        elif mode == "essthin":
            third = size//3
            maze[third, :size-1] = 1
            maze[2*third, 1:size] = 1
        return maze

    def generate_maze(self, maze: np.ndarray, iter_num: int = 10) -> np.ndarray:
        """
        This method generates maze from given template maze.
        It adds random walls to the maze and checks if the maze is solvable.
        If it is not solvable, it reverses the step adds +1 to non_solvable counter.
        It continues until non_solvable counter reaches iter_num.
        """
        incident_matrix = self.incident(maze)
        non_solvable = 0
        height, width = maze.shape
        cor, path = self.shortest_check(incident_matrix)
        numbers = np.random.choice(range(1, width**2), width**2 - 1, replace=False)
        i = 0
        while non_solvable < iter_num and i < width**2 - 1:
            num = numbers[i]
            row = num // width
            column = num % width
            maze[row, column] = 1
            incident_matrix_copy = incident_matrix[num, :].copy()
            incident_matrix[num, :] = 0
            incident_matrix[:, num] = 0
            if num in path:
                cor, path_new = self.shortest_check(incident_matrix)
                if not cor:
                    non_solvable += 1
                    incident_matrix[num, :] = incident_matrix_copy
                    incident_matrix[:, num] = incident_matrix_copy
                    maze[row, column] = 0
                else:
                    path = path_new
            i += 1
        return maze

    def incident(self, maze: np.ndarray) -> np.ndarray:
        """
        This method creates incident matrix from given maze.
        It creates a graph from the maze and returns its incident matrix.
        """
        height, width = maze.shape
        incident_matrix = np.zeros((height*width, height*width))
        for i in range(height):
            for j in range(width):
                if not maze[i, j]:
                    incident_row = i * width + j
                    if i != height - 1 and not maze[i+1, j]:
                        incident_column = (i + 1) * width + j
                        incident_matrix[incident_row, incident_column] = 1
                        incident_matrix[incident_column, incident_row] = 1
                    if j != width - 1 and not maze[i, j+1]:
                        incident_column = i * width + j + 1
                        incident_matrix[incident_row, incident_column] = 1
                        incident_matrix[incident_column, incident_row] = 1
        return incident_matrix

    def find_shortest_path(self, incident_matrix: np.ndarray, start_node: int = 0, end_node: int | None = None) -> list[int]:
        """
        This method finds the shortest path from start_node to end_node in the maze.
        It uses BFS algorithm.
        Shortest path is returned as a list of nodes.
        """
        if end_node is None:
            end_node = incident_matrix.shape[0] - 1
        visited = [False] * incident_matrix.shape[0]
        visited[start_node] = True
        queue = [start_node]
        list_of_paths = [[start_node]]
        while queue:
            current = queue.pop(0)
            path = list_of_paths.pop(0)
            if current == end_node:
                return path
            for i in range(incident_matrix.shape[0]):
                if not visited[i] and incident_matrix[current, i]:
                    visited[i] = True
                    queue.append(i)
                    list_of_paths.append(path + [i])
        return []

    def shortest_check(self, incident_matrix: np.ndarray, start_node: int = 0, end_node: int | None = None) -> tuple[bool, list[int]]:
        """
        This method checks if there is a path from start_node to end_node in the maze.
        It uses BFS algorithm.
        If there is a path, it returns True, otherwise False.
        """
        if end_node is None:
            end_node = incident_matrix.shape[0] - 1
        visited = [False] * incident_matrix.shape[0]
        visited[start_node] = True
        queue = [start_node]
        list_of_paths = [[start_node]]
        non_zero = [i for i, line in enumerate(incident_matrix) if np.any(line)]
        while queue:
            current = queue.pop(0)
            path = list_of_paths.pop(0)
            if current == end_node:
                return True, path
            for i in non_zero:
                if not visited[i] and incident_matrix[current, i]:
                    visited[i] = True
                    queue.append(i)
                    list_of_paths.append(path + [i])
        return False, []

    def add_path_to_maze(self, maze: np.ndarray, path: list[int]) -> np.ndarray:
        """
        This method adds the shortest path to the maze.
        It returns the maze with the path marked with different value.
        """
        maze = maze.astype(int)
        width = maze.shape[1]
        for i in path:
            row = i // width
            column = i % width
            maze[row, column] = 2
        return maze

    def plot_maze(self, maze: np.ndarray) -> None:
        """
        This method plots the maze using the matplotlib.
        It uses white color for empty space,
        black for walls and red for the path.
        Axis are turned off.
        """
        plt.imshow(maze, cmap=ListedColormap(['white', 'black', 'red']))
        plt.axis('off')
        plt.show()

    def solve_maze_file(self, file_name: str) -> None:
        """
        This method solves the maze from the given file.
        It loads the maze, creates incident matrix. 
        Finds the shortest path and adds it to the maze.
        Finally, it plots the maze.
        It uses the methods defined in the class Maze.
        """
        maze = self.load_maze(file_name)
        incident_matrix = self.incident(maze)
        path = self.find_shortest_path(incident_matrix)
        maze = self.add_path_to_maze(maze, path)
        self.plot_maze(maze)

    def solve_maze_generate(self, size: int, mode: Literal['empty', 'slalom', 'ess'], iter_num: int) -> None:
        """
        This method generates and solves the maze.
        It generates the maze template, creates incident matrix.
        Finds the shortest path and adds it to the maze.
        Finally, it plots the maze.
        It uses the methods defined in the class Maze.
        """
        maze = self.generate_temple(size, mode)
        maze = self.generate_maze(maze, iter_num)
        incident_matrix = self.incident(maze)
        path = self.find_shortest_path(incident_matrix)
        maze = self.add_path_to_maze(maze, path)
        self.plot_maze(maze)
