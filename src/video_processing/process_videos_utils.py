def normalize_point(point_to_normalize, avg_dist, avg_point):
        nx = point_to_normalize[0] - avg_point[0]
        nx = nx / avg_dist
        ny = point_to_normalize[1] - avg_point[1]
        ny = ny / avg_dist

        return [nx,ny]