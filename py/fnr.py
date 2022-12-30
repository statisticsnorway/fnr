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
    Docstring TBA
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

        # Setting up instance variables
        self.__year_from = from_year
        self.__year_to = to_year
        self.__aggregations = aggregations
        self.__regions = [x.lower() for x in regions if x.lower() != 'hele_landet']
        self.__data_path = data_path
        self.__catalogue_path = catalogue_path
        self.__variables = VARIABLES

        # Fill in empty lists and dicts if not supplied by user
        if self.__aggregations.get('lists') is None:
            self.__aggregations = {**self.__aggregations, **{'lists': []}}
        if self.__aggregations.get('mappings') is None:
            self.__aggregations = {**self.__aggregations, **{'mappings': {}}}

        # Run setup method and store data as private DataFrame
        self.__df = self.__setup_class(self.__year_from, self.__year_to, self.__aggregations)

    @property
    def year_from(self):
        return self.__year_from

    @property
    def year_to(self):
        return self.__year_to

    @property
    def regions(self):
        return self.__regions

    @property
    def data_path(self):
        return self.__data_path

    @property
    def catalogue_path(self):
        return self.__catalogue_path

    @property
    def df(self):
        return self.__df

    @regions.setter
    def regions(self, regions):
        self.__regions = regions

    @data_path.setter
    def data_path(self, path):
        if isinstance(path, str) is False:
            raise IOError('Path must be string')
        if os.path.exists(path):
            self.__data_path = path
        else:
            raise IOError('Path {} does not exist'.format(path))

    @catalogue_path.setter
    def catalogue_path(self, path):
        if isinstance(path, str) is False:
            raise IOError('Path must be string')
        if os.path.exists(path):
            self.__data_path = path
        else:
            raise IOError('Path {} does not exist'.format(path))

    # Method that returns all FNR data in one DataFrame in tidy format (long)
    def __setup_class(self, year_from, year_to, aggregations):
        print('Setting up class instance')

        # Getting data, concatenating, aggregating etc.
        df = self.__get_years(year_from, year_to).drop('varnr', axis=1)
        df = self.__fill_missing_regions(df)
        df = self.__set_aggregations_index(df, aggregations)
        df_aggregations = self.__make_aggregations_df(df, aggregations)
        df_aggregations_with_growth = self.__return_df_with_growth(df_aggregations)
        df_aggregations_with_growth_tidy = self.__make_tidy_df(df_aggregations_with_growth)

        print('Ready')

        return df_aggregations_with_growth_tidy

    # Method that gets FNR data for a several years and puts them in a DataFrame
    def __get_years(self, year_from, year_to):
        print('Reading data from {}\nLoading'.format(self.__data_path), end=' ')

        # Store years of data in list of DataFrames
        df_list = []
        for year in range(year_from, year_to+1):
            df_list.append(self.__get_year(year))

        print()

        # Concatenate (stack) list to single DataFrame
        df = pd.concat(df_list)

        # Set all variable names and object type variables to lowecase
        df.columns = [x.lower() for x in df.columns]
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.lower()

        return df

    # Method that gets FNR data for a single year and puts it in a DataFrame
    def __get_year(self, year):
        print(str(year), end=' ')

        df_list = []
        for var in self.__variables:
            df_list.append(pd.read_sas(''.join([self.__data_path, '_'.join(['fylke', var, str(year)]), '.sas7bdat']), encoding='iso-8859-1'))

        df = pd.concat(df_list)
        df = df.assign(**{'årgang': pd.Period(value=year, freq='A')})

        return df

    # Method that returns a DataFrame where regions are filled in according to region reform of 2018 and 2020
    def __fill_missing_regions(self, df):
        # Make sure DataFrame has all regions by
        df_fill = pd.concat([df.copy(deep=True), pd.DataFrame(columns=['hele_landet']+self.__regions, dtype=np.float64)])

        # Fill in regions according to regional reform of 2018 and 2020
        if 'f30' in df_fill.columns:
            df_fill['f30'] = np.where(df_fill['f30'].isnull(), df_fill['f01']+df_fill['f02']+df_fill['f06'], df_fill['f30'])
        if 'f34' in df_fill.columns:
            df_fill['f34'] = np.where(df_fill['f34'].isnull(), df_fill['f04']+df_fill['f05'], df_fill['f34'])
        if 'f38' in df_fill.columns:
            df_fill['f38'] = np.where(df_fill['f38'].isnull(), df_fill['f07']+df_fill['f08'], df_fill['f38'])
        if 'f42' in df_fill.columns:
            df_fill['f42'] = np.where(df_fill['f42'].isnull(), df_fill['f09']+df_fill['f10'], df_fill['f42'])
        if 'f46' in df_fill.columns:
            df_fill['f46'] = np.where(df_fill['f46'].isnull(), df_fill['f12']+df_fill['f14'], df_fill['f46'])
        if 'f50' in df_fill.columns:
            df_fill['f50'] = np.where(df_fill['f50'].isnull(), df_fill['f16']+df_fill['f17'], df_fill['f50'])
        if 'f54' in df_fill.columns:
            df_fill['f54'] = np.where(df_fill['f54'].isnull(), df_fill['f19']+df_fill['f20'], df_fill['f54'])

        return df_fill

    # Method that returns a DataFrame with MultiIndex
    def __set_aggregations_index(self, df, aggregations):
        print('Generating aggregations')

        df_index = df.copy(deep=True)

        for aggregation in aggregations.get('lists'):
            df_index = df_index.assign(**{aggregation: df['naering'].map(self.__make_aggregation_mapping(aggregation))})

        for aggregation in aggregations.get('mappings').keys():
            df_index = df_index.assign(**{aggregation: df['naering'].map({x.lower(): aggregation.lower() for x in aggregations.get('mappings')[aggregation]})})

        return df_index.set_index(['årgang', 'nr_variabler', *aggregations.get('lists'), *aggregations.get('mappings'), 'naering'])

    # Method that generates mapping from NR-næring to aggregation
    def __make_aggregation_mapping(self, aggregation):
        df = pd.read_sas(''.join([self.__catalogue_path, 'naering.sas7bdat']), encoding='iso-8859-1')

        return dict(zip(df['naering'].str.lower(), df[aggregation].str.lower()))

    # Method that returns DataFrame for every aggregate in aggregations
    def __make_aggregations_df(self, df, aggregations):
        print('Generating aggregated DataFrame')

        df_list = []
        for aggregation in ['naering',
                            *aggregations.get('lists'),
                            *aggregations.get('mappings').keys()]:
            df_list.append(
                            self.__return_aggregation_df(df, aggregation)
                            .reset_index()
                            .rename(columns={aggregation: 'aggregat'})
                            .set_index(['årgang', 'nr_variabler',
                                        'aggregering', 'aggregat'])
                          )

        return pd.concat(df_list)

    # Method that returns DataFrame aggregated according to aggregation
    def __return_aggregation_df(self, df, aggregation):
        df_aggregation = df.groupby(['årgang', 'nr_variabler', aggregation], dropna=False).sum(min_count=1)
        df_aggregation = df_aggregation.assign(**{'aggregering': aggregation.lower()})
        df_aggregation = df_aggregation[df_aggregation.index.get_level_values(aggregation).isnull() == False]

        return df_aggregation

    # Method that returns DataFrame with volume growth rates
    def __return_df_with_growth(self, df):
        print('Generating volume growth')

        df_vr = df[df.index.get_level_values('nr_variabler') == 'bnp'].droplevel('nr_variabler').sort_index()
        df_fp = df[df.index.get_level_values('nr_variabler') == 'bnpf'].droplevel('nr_variabler').sort_index()
        df_vl = (
            100*(df_fp-df_vr.groupby(['aggregat', 'aggregering']).shift(1))
            .divide(df_vr.groupby(['aggregat', 'aggregering']).shift(1))
            .assign(**{'nr_variabler': 'vlp'})
            .reset_index()
            .set_index(['årgang', 'nr_variabler', 'aggregering', 'aggregat'])
        )

        return pd.concat([df, df_vl.reset_index().set_index(['årgang', 'nr_variabler', 'aggregering', 'aggregat'])])

    # Method that takes wide DataFrame and transposes it to long (tidy format)
    def __make_tidy_df(self, df):
        df_tidy = (
            df
            .reset_index()
            .melt(id_vars=['årgang', 'nr_variabler', 'aggregering', 'aggregat'],
                  var_name='fylke', value_name='verdi')
            .set_index(['årgang', 'nr_variabler', 'aggregering', 'aggregat', 'fylke'])
        )

        return df_tidy

    # Method that adds data to class DataFrame
    def add_year(self, year):
        print('Adding year to class instance')

        if year == self.__year_to+1:
            # Getting data, concatenating, aggregating etc.
            df = self.__get_years(year, year).drop('varnr', axis=1)
            df = self.__fill_missing_regions(df)
            df = self.__set_aggregations_index(df, self.__aggregations)
            df_aggregations = self.__make_aggregations_df(df, self.__aggregations)
            df_aggregations_with_growth = self.__return_df_with_growth(df_aggregations)
            df_aggregations_with_growth_tidy = self.__make_tidy_df(df_aggregations_with_growth)

            # Storing new data to DataFrame and updating to_year
            self.__df = pd.concat([self.__df, df_aggregations_with_growth_tidy])
            self.__year_to = year

            print('Ready')
        else:
            warnings.warn('Cannot add year {} when to_year is {}'
                          .format(str(year), str(self.to_year)))

    # Method that returns a style opbject with selecte variables
    def return_selection(self, aggregation: str, years=None,
                         variables=None, aggregates=None,
                         regions=None, **kwargs) -> pd.DataFrame.style:
        """
        Method that returns a style object for selected aggregation, year(s), variable(s), aggregate(s), and region(s)
        Optional arguments (**kwargs) are:
            * wide_by: transpose to wide by selected variable
            * columns: list according to which columns will be sorted
            * sort_by: sort by selected variable or variables
            * round_to: round values to chosen number of decimals

        Example of use (assuming there's an class instance called 'fnr':
            * fnr.return_selection('pubagg', [2019,2020], ['bnp'], ['2x35', '2x41_43'], ['f30', 'f03', 'f34'], wide_by='fylke', round_to=2)
            * fnr.return_selection('naering', [2018,2019,2020], ['vlp'], ['23720'], ['f30', 'f03', 'f34'], sort_by=['årgang', 'nr_variabler'])

        The DataFrame underlying the style object may be retreived using return_selection().data (that is appending the statement 'data')
        """

        # Select all if years, aggregates or regions is None or empty list
        if years in (None, []):
            years = (
                self.__df
                .index
                .get_level_values('årgang')
                .year
                .unique()
                .to_list()
            )
        if variables in (None, []):
            variables = (
                self.__df
                .index
                .get_level_values('nr_variabler')
                .unique()
                .to_list()
            )
        if aggregates in (None, []):
            aggregates = (
                self.__df[self.df.index.get_level_values('aggregering') == aggregation]
                .index
                .get_level_values('aggregat')
                .unique()
                .to_list()
            )
        if regions in (None, []):
            regions = ['hele_landet']+(
                self.__df
                .index
                .get_level_values('fylke')
                .unique()
                .to_list()
            )

        # Make condition for selection and DataFrame thats satisfies condition
        condition = (
            (self.__df.index.get_level_values('årgang').year.isin(years)) &
            (self.__df.index.get_level_values('nr_variabler').isin([x.lower() for x in variables])) &
            (self.__df.index.get_level_values('aggregering') == aggregation.lower()) &
            (self.__df.index.get_level_values('aggregat').isin([x.lower() for x in aggregates])) & 
            (self.__df.index.get_level_values('fylke').isin([x.lower() for x in regions]))
        )

        df = self.__df[condition]

        # Reshape DataFrame to wide by chosen variable, if any
        if 'wide_by' in kwargs.keys():
            wide_by = kwargs.get('wide_by')
            df = (
                df
                .reset_index()
                .pivot(index=[x for x in ['årgang', 'nr_variabler', 'aggregat', 'fylke'] if x != wide_by],
                       columns=[wide_by], values='verdi')
            )

        # Order columns according to list, if any
        if 'columns' in kwargs.keys():
            df = df.reindex(columns=[x.lower() for x in kwargs.get('columns')])

        # Sort data by chosen variable or list of variables, if requested
        if 'sort_by' in kwargs.keys():
            try:
                df = df.sort_values(kwargs.get('sort_by'))
            except KeyError:
                raise KeyError('Cannot sort by {}.'
                               'Check that variables exist and are not in wide_by'
                               .format(kwargs.get('sort_by')))

        # Return results as style object, rounded if requested
        if 'round_to' in kwargs.keys():
            return df.style.format(precision=kwargs.get('round_to'), decimal=',')
        else:
            return df.style

    # Method that ...
    def supress_data():
        pass

    # Method that ...
    def to_statbank():
        pass