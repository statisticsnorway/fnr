############################
# Author: Magnus Helliesen #
############################

import numpy as np
import pandas as pd
import warnings
import os

VARIABLES = ['prod', 'pin', 'bnp', 'bnpf', 'brin', 'lkost', 'syss']


class fnr_class:
    """
    Class that loads and stores data for the Regional accounts (RA).
    The class reads SAS-datasets (sas7bdat-files) containing data and a catalogue containing aggregations.
    It aggregates all the data according to user input and calculates colume growth rates.
    The class contains methods to effortlessly display data in a user friendly format.

    An instance (fnr) of the class is set up using
        * fnr = fnr_class(year_from, year_to, aggregations, regions, data_path, catalogue_path)

    where
        * aggregations is a dict containing the keys 'lists' and/or 'mappings' (or neither), where
            - 'lists' contains a list of aggregations in the catalogue to aggregated over
            - 'mappings contains a dict where each key contains some aggregation
        * regions is a list of regions
        * data_path is the path to the SAS datasets containing the data
        * catalogue_path is the path to the SAS datasets containing the catalogue

    Example of aggregations dict: {'list': ['a38np', 'pubagg'], 'mappings': {'23ind': ['23101', '23102', '23103', '23104',...]}}
    """

    def __init__(self, from_year, to_year, aggregations, regions, data_path, catalogue_path):
        # Check that paths to data and catalogues exist, the latter only if used
        if isinstance(data_path, str) is False:
            raise IOError('Data path must be string')
        if os.path.exists(data_path) is False:
            raise IOError('Path {} does not exist'.format(data_path))
        if isinstance(catalogue_path, str) is False and aggregations.get('lists') is not None:
            raise IOError('Catalogue path must be string')
        if os.path.exists(catalogue_path) is False and aggregations.get('lists') is not None:
            raise IOError('Path {} does not exist'.format(catalogue_path))

        # Set up instance variables
        self._year_from = from_year
        self._year_to = to_year
        self._aggregations = aggregations
        self._regions = [x.lower() for x in regions if x.lower() != 'hele_landet']
        self._data_path = data_path
        self._catalogue_path = catalogue_path
        self._variables = VARIABLES

        # Fill in empty lists and dicts if not supplied by user
        if self._aggregations.get('lists') is None:
            self._aggregations = {**self._aggregations, **{'lists': []}}
        if self._aggregations.get('mappings') is None:
            self._aggregations = {**self._aggregations, **{'mappings': {}}}

        # Run setup method and store data as private DataFrame
        self._df_untidy, self._df, self._agg_dict = self._setup_class(self._year_from, self._year_to, self._aggregations)

    @property
    def year_from(self):
        return self._year_from

    @property
    def year_to(self):
        return self._year_to

    @property
    def regions(self):
        return self._regions

    @property
    def data_path(self):
        return self._data_path

    @property
    def catalogue_path(self):
        return self._catalogue_path

    @property
    def df(self):
        return self._df

    @property
    def agg_dict(self):
        return self._agg_dict

    @regions.setter
    def regions(self, regions):
        self._regions = regions

    @data_path.setter
    def data_path(self, path):
        if isinstance(path, str) is False:
            raise IOError('Path must be string')
        if os.path.exists(path):
            self._data_path = path
        else:
            raise IOError('Path {} does not exist'.format(path))

    @catalogue_path.setter
    def catalogue_path(self, path):
        if isinstance(path, str) is False:
            raise IOError('Path must be string')
        if os.path.exists(path):
            self._data_path = path
        else:
            raise IOError('Path {} does not exist'.format(path))

    # Method that returns all FNR data in one DataFrame in tidy format (long)
    def _setup_class(self, year_from, year_to, aggregations):
        print('Setting up class instance')

        # Getting data, concatenating, aggregating etc.
        df = self._get_years(year_from, year_to).drop('varnr', axis=1)
        df = self._fill_missing_regions(df)
        df, agg_dict = self._set_aggregations_index(df, aggregations)
        df_aggregations = self._make_aggregations_df(df, aggregations)
        df_aggregations_with_growth = self._return_df_with_growth(df_aggregations)
        df_aggregations_with_growth_tidy = self._make_tidy_df(df_aggregations_with_growth)

        print('Ready')

        return df_aggregations_with_growth, df_aggregations_with_growth_tidy, agg_dict

    # Method that gets FNR data for a several years and puts them in a DataFrame
    def _get_years(self, year_from, year_to):
        print('Reading data from {}\nLoading'.format(self._data_path), end=' ')

        # Store years of data in list of DataFrames
        df_list = []
        for year in range(year_from, year_to+1):
            df_list.append(self._get_year(year))

        print()

        # Concatenate (stack) list to single DataFrame
        df = pd.concat(df_list)

        # Set all variable names and object type variables to lowercase
        df.columns = [x.lower() for x in df.columns]
        for col in df.select_dtypes(include=['object']).columns:
            df = df.assign(**{col: lambda df: df[col].str.lower()})

        return df

    # Method that gets FNR data for a single year and puts it in a DataFrame
    def _get_year(self, year):
        print(str(year), end=' ')

        df_list = []
        for var in self._variables:
            df_list.append(pd.read_sas(''.join([self._data_path, '_'.join(['fylke', var, str(year)]), '.sas7bdat']), encoding='iso-8859-1'))

        df = pd.concat(df_list)
        df = df.assign(**{'??rgang': pd.Period(value=year, freq='A')})

        return df

    # Method that returns a DataFrame where regions are filled in according to region reform of 2018 and 2020
    def _fill_missing_regions(self, df):
        # Make sure DataFrame has all regions by
        df_fill = pd.concat([df.copy(deep=True), pd.DataFrame(columns=['hele_landet']+self._regions, dtype=np.float64)])

        # Fill in regions according to regional reform of 2018 and 2020
        df_fill = df_fill.assign(**{'f30': np.where(df_fill['f30'].isnull(), df_fill['f01']+df_fill['f02']+df_fill['f06'], df_fill['f30'])})
        df_fill = df_fill.assign(**{'f34': np.where(df_fill['f34'].isnull(), df_fill['f04']+df_fill['f05'], df_fill['f34'])})
        df_fill = df_fill.assign(**{'f38': np.where(df_fill['f38'].isnull(), df_fill['f07']+df_fill['f08'], df_fill['f38'])})
        df_fill = df_fill.assign(**{'f42': np.where(df_fill['f42'].isnull(), df_fill['f09']+df_fill['f10'], df_fill['f42'])})
        df_fill = df_fill.assign(**{'f46': np.where(df_fill['f46'].isnull(), df_fill['f12']+df_fill['f14'], df_fill['f46'])})
        df_fill = df_fill.assign(**{'f50': np.where(df_fill['f50'].isnull(), df_fill['f16']+df_fill['f17'], df_fill['f50'])})
        df_fill = df_fill.assign(**{'f54': np.where(df_fill['f54'].isnull(), df_fill['f19']+df_fill['f20'], df_fill['f54'])})

        return df_fill

    # Method that returns a DataFrame with MultiIndex
    def _set_aggregations_index(self, df, aggregations):
        print('Generating aggregations')

        df_index = df.copy(deep=True)
        agg_dict = {}

        # Make variables mapping from naering to aggregations in lists
        for aggregation in aggregations.get('lists'):
            mapping1, mapping2 = self._make_aggregation_mapping(aggregation)
            df_index = df_index.assign(**{aggregation: df['naering'].map(
                mapping1
            )})

            agg_dict = {**agg_dict, **{aggregation: mapping2}}

        # Make variables mapping from naering to aggregations in mappings
        for aggregation in aggregations.get('mappings').keys():
            df_index = df_index.assign(**{aggregation: df['naering'].map(
                {x.lower(): aggregation.lower() for x in aggregations.get('mappings').get(aggregation)}
            )})

        agg_dict = {**agg_dict, **aggregations.get('mappings')}

        return df_index.set_index(['??rgang', 'nr_variabler', *aggregations.get('lists'), *aggregations.get('mappings'), 'naering']), agg_dict

    # Method that generates mapping from NR-n??ring to aggregation and the other way around
    def _make_aggregation_mapping(self, aggregation):
        df = pd.read_sas(''.join([self._catalogue_path, 'naering.sas7bdat']), encoding='iso-8859-1')
        df = df[(df['naering'].str.startswith('2')) | (df['naering'].str.startswith('8'))]
        df = df.assign(**{aggregation: df[aggregation].str.lower()})

        return dict(zip(df['naering'].str.lower(), df[aggregation].str.lower())), dict(df.groupby(aggregation)['naering'].apply(list))

    # Method that returns DataFrame for every aggregate in aggregations
    def _make_aggregations_df(self, df, aggregations):
        print('Generating aggregated DataFrame')

        df_list = []
        for aggregation in ['naering',
                            *aggregations.get('lists'),
                            *aggregations.get('mappings').keys()]:
            df_list.append(
                            self._return_aggregation_df(df, aggregation)
                            .reset_index()
                            .rename(columns={aggregation: 'aggregat'})
                            .set_index(['??rgang', 'nr_variabler',
                                        'aggregering', 'aggregat'])
                          )

        return pd.concat(df_list)

    # Method that returns DataFrame aggregated according to aggregation
    def _return_aggregation_df(self, df, aggregation):
        df_aggregation = df.groupby(['??rgang', 'nr_variabler', aggregation], dropna=False).sum(min_count=1)
        df_aggregation = df_aggregation.assign(**{'aggregering': aggregation.lower()})
        df_aggregation = df_aggregation[df_aggregation.index.get_level_values(aggregation).isnull() == False]

        return df_aggregation

    # Method that returns DataFrame with volume growth rates
    def _return_df_with_growth(self, df):
        print('Generating volume growth')

        df_vr = df[df.index.get_level_values('nr_variabler') == 'bnp'].droplevel('nr_variabler').sort_index()
        df_fp = df[df.index.get_level_values('nr_variabler') == 'bnpf'].droplevel('nr_variabler').sort_index()
        df_vl = (
            100*(df_fp-df_vr.groupby(['aggregat', 'aggregering']).shift(1))
            .divide(df_vr.groupby(['aggregat', 'aggregering']).shift(1))
            .assign(**{'nr_variabler': 'vlp'})
            .reset_index()
            .set_index(['??rgang', 'nr_variabler', 'aggregering', 'aggregat'])
        )

        return pd.concat([df, df_vl.reset_index().set_index(['??rgang', 'nr_variabler', 'aggregering', 'aggregat'])])

    # Method that takes wide DataFrame and transposes it to long (tidy format)
    def _make_tidy_df(self, df):
        df_tidy = (
            df
            .reset_index()
            .melt(id_vars=['??rgang', 'nr_variabler', 'aggregering', 'aggregat'],
                  var_name='fylke', value_name='verdi')
            .set_index(['??rgang', 'nr_variabler', 'aggregering', 'aggregat', 'fylke'])
            .assign(**{'prikke': False})
        )

        return df_tidy

    # Method that adds data to class DataFrame
    def add_year(self, year):
        print('Adding year to class instance')

        if year == self._year_to+1:
            # Getting data, concatenating, aggregating etc.
            df = self._get_years(year, year).drop('varnr', axis=1)
            df = self._fill_missing_regions(df)
            df, _ = self._set_aggregations_index(df, self._aggregations)
            df_aggregations = self._make_aggregations_df(df, self._aggregations)
            df_aggregations = pd.concat([self._df_untidy[self._df_untidy.index.get_level_values('nr_variabler') != 'vlp'], df_aggregations])
            df_aggregations_with_growth = self._return_df_with_growth(df_aggregations)
            df_aggregations_with_growth = df_aggregations_with_growth[df_aggregations_with_growth.index.get_level_values('??rgang').year == year]
            df_aggregations_with_growth_tidy = self._make_tidy_df(df_aggregations_with_growth)

            # Storing new data to DataFrame and updating to_year
            self._df = pd.concat([self._df, df_aggregations_with_growth_tidy])
            self._year_to = year

            print('Ready')
        else:
            warnings.warn('Cannot add year {} when to_year is {}'
                          .format(str(year), str(self.year_to)))

    # Method that returns a style opbject with selecte variables
    def return_selection(self, aggregation: str, years=None,
                         variables=None, aggregates=None,
                         regions=None, **kwargs) -> pd.DataFrame.style:
        """
        Method that returns a style object for selected aggregation, year(s), variable(s), aggregate(s), and region(s).
        Optional arguments (**kwargs) are:
            * wide_by: transpose to wide by selected variable
            * columns: list according to which columns will be sorted
            * sort_by: sort by selected variable or variables
            * round_to: round values to chosen number of decimals
            * first_diff: calculate first differences False/True
            * suppress: show NaN for suppressed observations False/True

        Example of use (assuming there's an class instance called 'fnr':
            * fnr.return_selection('pubagg', [2019,2020], ['bnp'], ['2x35', '2x41_43'], ['f30', 'f03', 'f34'], wide_by='fylke', round_to=2)
            * fnr.return_selection('naering', [2018,2019,2020], ['vlp'], ['23720'], ['f30', 'f03', 'f34'], sort_by=['??rgang', 'nr_variabler'])

        The DataFrame underlying the style object may be retreived using return_selection().data (that is appending the statement 'data')
        """

        # Make sure that input is a list
        years = fnr_class._flatten_list([years])
        variables = fnr_class._flatten_list([variables])
        aggregates = fnr_class._flatten_list([aggregates])
        regions = fnr_class._flatten_list([regions])

        # Select all if years, aggregates or regions is None or empty list
        if years in ([None], []):
            years = (
                self._df
                .index
                .get_level_values('??rgang')
                .year
                .unique()
                .to_list()
            )
        if variables in ([None], []):
            variables = (
                self._df
                .index
                .get_level_values('nr_variabler')
                .unique()
                .to_list()
            )
        if aggregates in ([None], []):
            aggregates = (
                self._df[self.df.index.get_level_values('aggregering') == aggregation]
                .index
                .get_level_values('aggregat')
                .unique()
                .to_list()
            )
        if regions in ([None], []):
            regions = ['hele_landet']+(
                self._df
                .index
                .get_level_values('fylke')
                .unique()
                .to_list()
            )

        # Make condition for selection and DataFrame thats satisfies condition
        condition = (
            (self._df.index.get_level_values('??rgang').year.isin(years)) &
            (self._df.index.get_level_values('nr_variabler').isin([x.lower() for x in variables])) &
            (self._df.index.get_level_values('aggregering') == aggregation.lower()) &
            (self._df.index.get_level_values('aggregat').isin([x.lower() for x in aggregates])) & 
            (self._df.index.get_level_values('fylke').isin([x.lower() for x in regions]))
        )

        # Suppress data (set to NaN) if chosen by user
        if kwargs.get('suppress') is True:
            df = self._df.assign(**{'verdi': lambda df: [x if y is False else np.nan for x, y in zip(df['verdi'], df['prikke'])]})[['verdi']]
        else:
            df = self._df[['verdi']]

        # Return first differences if chosen by user
        if kwargs.get('first_diff') is True:
            df = (
                df
                .groupby(['nr_variabler', 'aggregering', 'aggregat', 'fylke'])
                .diff(1)
            )[condition]
        else:
            df = df[condition]

        # Reshape DataFrame to wide by chosen variable, if any
        if 'wide_by' in kwargs.keys():
            wide_by = kwargs.get('wide_by')
            df = (
                df
                .reset_index()
                .pivot(index=[x for x in ['??rgang', 'nr_variabler', 'aggregat', 'fylke'] if x != wide_by],
                       columns=[wide_by], values='verdi')
            )

        # Order columns according to list, if any
        if 'columns' in kwargs.keys():
            df = df.reindex(columns=[x.lower() for x in kwargs.get('columns')])

        # Sort data by chosen variable or list of variables, if requested
        if 'sort_by' in kwargs.keys():
            try:
                df = df.sort_values(kwargs.get('sort_by'), ascending=False)
            except KeyError:
                raise KeyError('Cannot sort by {}.'
                               'Check that variables exist and are not in wide_by'
                               .format(kwargs.get('sort_by')))

        # Return results as style object, rounded if requested
        if 'round_to' in kwargs.keys():
            return df.style.format(precision=kwargs.get('round_to'), decimal=',')
        else:
            return df.style

    # Method that suppresses data according to dict {year: [[aggreagte, region]]
    def suppress_data(self, to_be_suppressed):
        """
        Method that suppresses data according to dict with years as keys and list/tuple of lists/tuples with [aggregate, region]'s to be suppressed.

        Example of use (assuming there's an class instance called 'fnr':
            * fnr.suppress_data({2019: [['2x90_97', 'f54'], ['2x85', 'f30']], 2020: [['2x90_97', 'f54'], ['2x85', 'f30']]})
        """

        df = self._df.copy(deep=True)

        # Loop over keys (years) and lists ([aggreagte, region]) and set 'prikke' to True
        for year in to_be_suppressed.keys():
            for [aggregate, region] in to_be_suppressed.get(year):
                df = df.assign(**{'prikke': lambda df: (
                    (df.index.get_level_values('??rgang').year == year) &
                    (df.index.get_level_values('aggregat') == aggregate.lower()) &
                    (df.index.get_level_values('fylke') == region.lower()) |
                    (df['prikke'])
                )})

        self._df = df

    # Method that ...
    def to_statbank():
        pass

    # Recursive function that flattens arbitrarily nested in_list
    @staticmethod
    def _flatten_list(in_list):
        if (isinstance(in_list, list) or isinstance(in_list, tuple)) is False:
            raise ValueError('Input must be list')

        outer_list = []
        for x in in_list:
            inner_list = []
            if isinstance(x, list) or isinstance(x, tuple):
                inner_list.extend(fnr_class._flatten_list(x))
            else:
                inner_list.append(x)
            outer_list.extend(inner_list)

        return outer_list