"""A Yelp-powered Restaurant Recommendation Program"""

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location.
    If multiple centroids are equally close, return the first one."""
    return min([i for i in centroids], key=lambda i: distance(location, i))

def group_by_first(pairs):
    """Return a list of lists that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key."""
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    return group_by_first([[find_closest(restaurant_location(i), centroids), i] for i in restaurants])


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    list = [restaurant_location(i) for i in cluster]
    x, y = [i[0] for i in list], [i[1] for i in list]
    return [mean(x), mean(y)]


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k,
    old_centroids, n = [], 0
    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        a, j = group_by_centroid(restaurants, centroids), 0
        for i in a:
            centroids[j] = find_centroid(i)
            j += 1
        n += 1
    return centroids

def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model."""
    xs = [feature_fn(r) for r in restaurants]
    ys = [user_rating(user, restaurant_name(r)) for r in restaurants]
    xx, yy = sum([(i - mean(xs))**2 for i in xs]), sum([(i - mean(ys))**2 for i in ys])
    intermediate = zip([(i - mean(xs)) for i in xs], [(i - mean(ys)) for i in ys])
    xxyy = sum([i[0] * i[1] for i in intermediate])
    b = xxyy / xx
    a = mean(ys) - b * mean(xs)
    r_squared = xxyy**2 / (xx * yy)

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature."""
    reviewed = user_reviewed_restaurants(user, restaurants)
    func_pred = [find_predictor(user, reviewed, fn)[0] for fn in feature_fns]
    rsquared = [find_predictor(user, reviewed, fn)[1] for fn in feature_fns]
    k = max([i for i in list(range(len(rsquared)))], key=lambda i: rsquared[i])
    return func_pred[k]

def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based on a function from feature_fns."""
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    dict = {}
    for restaurant in restaurants:
        if restaurant in reviewed:
            rating = user_rating(user, restaurant_name(restaurant))
        else:
            rating = predictor(restaurant)
        dict[restaurant_name(restaurant)] = rating
    return dict

def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category."""
    total = []
    for restaurant in restaurants:
        if query in restaurant_categories(restaurant):
            total.append(restaurant)
    return total

def feature_set():
    """Return a sequence of feature functions."""
    return [lambda r: mean(restaurant_ratings(r)),
            restaurant_price,
            lambda r: len(restaurant_ratings(r)),
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
