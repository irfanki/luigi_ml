""" Rubikloud take home problem """
import ast
import luigi
import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    # TODO...
    def run(self):
        df_airline = pd.read_csv(self.tweet_file, engine='python')

        # Removing rows containing NAN in the tweet_coord column
        df_airline.dropna(subset=['tweet_coord'], inplace=True)

        # Removing rows containing '0.0, 0.0' in the tweet_coord column
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
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    # TODO...
    def requires(self):
        return CleanDataTask()

    def run(self):
        le = preprocessing.LabelEncoder()

        df_clean = pd.read_csv(CleanDataTask().output().path)
        df_cities = pd.read_csv(self.cities_file)

        # Converting list as string to list
        df_clean['tweet_coord'] = df_clean['tweet_coord'].apply(lambda x: ast.literal_eval(x))

        # We can consider dropping the duplicate values as well below is a code for the same.
        df_cities = df_cities.drop_duplicates(subset='name', keep="first")
        df_cities.reset_index(drop=True, inplace=True)

        # combining the latitude and longitude columns of the dataframe
        df_cities['latlong'] = df_cities.apply(lambda x: list([x['latitude'], x['longitude']]), axis=1)

        # Using the KDtree to locate the nearest city for the respective coordinate
        s2 = np.array(list(df_cities['latlong']))
        s1 = np.array(list(df_clean['tweet_coord']))
        kdtree = KDTree(s1)
        neighbours = kdtree.query(s2, k=1)

        airline_sentiment = []
        for i in neighbours[1]:
            airline_sentiment.append(df_clean['airline_sentiment'][i])

        # Creating dataframe containing the city name and airline sentiment for the respective city
        df_features = pd.DataFrame(list(zip(df_cities['name'], airline_sentiment)), columns=['city_name',
                                                                                             'airline_sentiment'])

        # As the data has imbalance so a fraction of excess label data has been randomly removed.
        df_features = df_features.drop(df_features[df_features['airline_sentiment'] == 'negative'].sample(frac=.54).index)
        df_features = df_features.drop(df_features[df_features['airline_sentiment'] == 'neutral'].sample(frac=.50).index)
        df_features.reset_index(drop=True, inplace=True)

        # Label and one hot encoding
        df_features['airline_sentiment'] = le.fit_transform(df_features['airline_sentiment'])
        df_features = pd.concat([df_features, pd.get_dummies(df_features['city_name'],
                                                             prefix='city_name', drop_first=True)], axis=1)
        df_features.drop(columns=['city_name'], inplace=True)

        with self.output().open('w') as f:
            # crate the final output
            df_features.to_csv(f, index=False)

    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    # TODO...
    def requires(self):
        return TrainingDataTask()

    def run(self):
        df_feature = pd.read_csv(TrainingDataTask().output().path)
        y = df_feature.pop('airline_sentiment')
        X = df_feature

        # model = GradientBoostingClassifier(random_state=1)
        model = LogisticRegression()
        model.fit(X, y)

        pickle.dump(model, open(self.output_file, 'wb'))

    def output(self):
        return luigi.LocalTarget(self.output_file)


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    # TODO...
    def requires(self):
        return TrainModelTask()

    def run(self):
        df_feature = pd.read_csv(TrainingDataTask().output().path)
        _ = df_feature.pop('airline_sentiment')
        X = df_feature

        # Reverse the pandas dummy to get the city name and remove the suffix
        name = X.idxmax(axis=1)
        city_name = list(map(lambda x: x.replace('city_name_', ''), name))
        loaded_model = pickle.load(open(TrainModelTask().output().path, 'rb'))

        # In the below code I am testing on the same dataset in which I trained on which I believe
        # is not a good idea however I did it as per my understanding of the requirement.We can do a train
        # test split as well.
        negative_probability = loaded_model.predict_proba(X)[:, 0]
        neutral_probability = loaded_model.predict_proba(X)[:, 1]
        positive_probability = loaded_model.predict_proba(X)[:, 2]

        df_score = pd.DataFrame(list(zip(city_name, negative_probability, neutral_probability, positive_probability)),
                                columns=['city_name', 'negative_probability', 'neutral_probability',
                                         'positive_probability'])

        df_score.sort_values('positive_probability', inplace=True, ascending=False)

        with self.output().open('w') as f:
            # crate the final output
            df_score.to_csv(f, index=False)

    def output(self):
        return luigi.LocalTarget(self.output_file)


if __name__ == "__main__":
    luigi.run()
    # luigi.build([CleanDataTask(default='airline_tweets.csv')], local_scheduler=True)
    # luigi.build([ScoreTask(default='airline_tweets.csv')], local_scheduler=True)
