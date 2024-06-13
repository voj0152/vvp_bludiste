import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Literal
from scipy.sparse import csr_matrix, lil_matrix


class Maze:
    """
    Class for creating and solving mazes.
    """

    def __init__(self):
        pass

    def load_maze(self, file_name: str) -> np.ndarray[np.bool_]:
        """
        This method loads a maze from a file stored in nd.array.
        """
        maze = np.loadtxt(file_name, delimiter=",", dtype=bool)
        return maze

    def generate_temple(
        self,
        size: int,
        mode: Literal['empty', 'slalom', 'ess', 'essthin'] = "empty"
    ) -> np.ndarray[np.bool_]:
        """
        This method generates maze templete with given size and mode.
        There are four modes: empty, slalom, ess, essthin.
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

    def generate_maze(
        self,
        maze: np.ndarray[np.bool_],
        iter_num: int | None = None
    ) -> np.ndarray[np.bool_]:
        """
        This method generates maze from given template maze.
        It adds random walls to the maze and checks if the maze is solvable.
        If it is not solvable, it reverses the step
        and adds +1 to non_solvable counter.
        It continues until non_solvable counter reaches iter_num
        or until all numbers are tried.
        """
        if iter_num is None:
            iter_num = maze.shape[0]
        incident_matrix = self.incident(maze)
        non_solvable = 0
        height, width = maze.shape
        numbers = np.random.choice(
            range(1, width**2 - 1), width**2 - 2, replace=False)
        i = 0
        while non_solvable < iter_num and i < numbers.size:
            num = numbers[i]
            row = num // width
            column = num % width
            maze[row, column] = 1
            indices = np.where(incident_matrix.indices == num)
            incident_matrix.data[indices] = 0
            if not self.shortest_check(incident_matrix):
                non_solvable += 1
                incident_matrix.data[indices] = 1
                maze[row, column] = 0
            i += 1
        return maze

    def shortest_check(
        self,
        incident_matrix: csr_matrix,
        start_node: int = 0,
        end_node: int | None = None
    ) -> bool:
        """
        This method checks if there is a path
        from start_node to end_node in the maze.
        It uses BFS algorithm implemented with matrix multiplication.
        If there is a path, it returns True otherwise False.
        """
        if end_node is None:
            end_node = incident_matrix.shape[0] - 1
        size = incident_matrix.shape[0]
        possible = np.zeros(size, dtype=bool)
        possible[start_node] = True
        for i in range(size - 1):
            possible = incident_matrix.dot(possible)
            if possible[end_node]:
                return True
        return False

    def incident(self, maze: np.ndarray[np.bool_]) -> csr_matrix:
        """
        This method creates incident matrix from given maze.
        It creates a graph from the maze and returns its incident matrix.
        """
        height, width = maze.shape
        incident_matrix = lil_matrix((height*width, height*width), dtype=bool)
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
        return csr_matrix(incident_matrix)

    def find_shortest_path(
        self,
        incident_matrix: csr_matrix,
        start_node: int = 0,
        end_node: int | None = None
    ) -> list[int]:
        """
        This method finds the shortest path
        from start_node to end_node in the maze.
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
            start = incident_matrix.indptr[current]
            end = incident_matrix.indptr[current + 1]
            for i in incident_matrix.indices[start:end]:
                if not visited[i]:
                    visited[i] = True
                    queue.append(i)
                    list_of_paths.append(path + [i])
        return []

    def add_path_to_maze(
        self,
        maze: np.ndarray[np.bool_],
        path: list[int]
    ) -> np.ndarray[np.int64]:
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

    def plot_maze(self, maze: np.ndarray[np.int64]) -> None:
        """
        This method plots the maze using the matplotlib.
        It uses white color for empty space,
        black for walls and red for the path.
        Axis are turned off.
        """
        plt.rcParams['figure.figsize'] = [3, 3]
        plt.figure(facecolor='black')
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

    def solve_maze_generate(
        self,
        size: int,
        mode: Literal['empty', 'slalom', 'ess', 'essthin'] = 'empty',
        iter_num: int | None = None
    ) -> None:
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
