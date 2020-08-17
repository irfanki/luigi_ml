""" Rubikloud take home problem """
import ast
import luigi
import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='clean_data.csv')

    # TODO...
    def run(self):
        df_airline = pd.read_csv(self.tweet_file, engine='python')
        # Dropping the irrelevant columns
        df_airline.drop(columns=['_unit_id', '_golden', '_unit_state', '_trusted_judgments',
                                 '_last_judgment_at',
                                 'airline_sentiment:confidence', 'negativereason',
                                 'negativereason:confidence', 'airline', 'airline_sentiment_gold',
                                 'name', 'negativereason_gold', 'retweet_count', 'text',
                                 'tweet_created', 'tweet_id', 'tweet_location', 'user_timezone'], inplace=True)

        # Cleaning the airline dataframe
        # Removing rows containing NAN
        df_airline.dropna(inplace=True)

        # Removing rows containing '0.0, 0.0'
        df_airline['tweet_coord'] = df_airline['tweet_coord'].astype(str)
        df_airline.drop(df_airline[df_airline.tweet_coord == '[0.0, 0.0]'].index, inplace=True)
        df_airline.reset_index(drop=True, inplace=True)

        with self.output().open('w') as f:
            # crate the final output
            df_airline.to_csv(f, index=False)

    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    # TODO...
    def requires(self):
        return CleanDataTask()

    def run(self):
        df_clean = pd.read_csv(CleanDataTask().output().path)
        # Converting list as string to list
        df_clean['tweet_coord'] = df_clean['tweet_coord'].apply(lambda x: ast.literal_eval(x))

        df_cities = pd.read_csv(self.cities_file)
        # combining the latitude and longitude columns of the dataframe
        df_cities['latlong'] = df_cities.apply(lambda x: list([x['latitude'], x['longitude']]), axis=1)

        # Using the KDtree to locate the nearest city for the respective coordinate
        s1 = np.array(list(df_cities['latlong']))
        s2 = np.array(list(df_clean['tweet_coord']))

        kdtree = KDTree(s1)
        neighbours = kdtree.query(s2, k=1)

        nearest_city = []
        for i in range(len(neighbours[1])):
            nearest_city.append(df_cities['name'][i])

        df_clean['nearest_city'] = nearest_city
        df_clean.drop(columns=['tweet_coord'], inplace=True)

        with self.output().open('w') as f:
            # crate the final output
            df_clean.to_csv(f, index=False)

    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='model.pkl')

    # TODO...
    def requires(self):
        return TrainingDataTask()

    def run(self):
        df_feature = pd.read_csv(TrainingDataTask().output().path)

        # As the data has imbalance so a fraction of excess label data has been randomly removed.
        df_feature = df_feature.drop(df_feature[df_feature['airline_sentiment'] == 'negative'].sample(frac=.78).index)
        df_feature = df_feature.drop(df_feature[df_feature['airline_sentiment'] == 'positive'].sample(frac=.15).index)
        df_feature.reset_index(drop=True, inplace=True)

        # One hot encoding the features
        X = pd.get_dummies(df_feature["nearest_city"], prefix='nearest_city', drop_first=True)

        # Labelencoding the target
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(df_feature['airline_sentiment'])

        model = GradientBoostingClassifier(random_state=1)
        model.fit(X, y)

        pickle.dump(model, open(self.output_file, 'wb'))

    def output(self):
        return luigi.LocalTarget(self.output_file)




# class ScoreTask(luigi.Task):
#     """ Uses the scored model to compute the sentiment for each city.
#
#         Output file should be a four column CSV with columns:
#         - city name
#         - negative probability
#         - neutral probability
#         - positive probability
#     """
#     tweet_file = luigi.Parameter()
#     output_file = luigi.Parameter(default='scores.csv')
#
#     # TODO...


if __name__ == "__main__":
    # luigi.run()
    # luigi.build([CleanDataTask()], local_scheduler=True)
    luigi.build([TrainModelTask()], local_scheduler=True)
