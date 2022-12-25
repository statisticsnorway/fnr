# Author: Magnus Helliesen
import numpy as np
import pandas as pd

DATA = '/ssb/stamme02/nasjregn/fnr_mr2022/wk24/'
CATALOGUES = '/ssb/stamme02/nasjregn/systemkataloger_mr2022/kat/'
VARS = ['prod', 'pin', 'bnp', 'bnpf', 'brin', 'lkost', 'syss']


class fnr_class:
    """
    Docstring TBA
    """

    def __init__(self, from_year, to_year, aggregations):
        self.__aggregations = aggregations

        # Fill in empty lists and dicts if not supplied by user
        if self.__aggregations.get('lists') is None:
            self.__aggregations = {**self.__aggregations, **{'lists': []}}
        if self.__aggregations.get('mappings') is None:
            self.__aggregations = {**self.__aggregations, **{'mappings': {}}}

        # Run setup method and store data as private DataFrame
        self.__df = self.__setup_class(from_year, to_year, self.__aggregations)
        print('Ready')

    @property
    def df(self):
        return self.__df

    # Function that returns all FNR data in one DataFrame
    def __setup_class(self, from_year, to_year, aggregations):
        df = self.__get_years(from_year, to_year).drop('varnr', axis=1)
        df = self.__fill_missing_regions(df)
        df = self.__set_aggregations_index(df, aggregations)
        df_aggregations = self.__make_aggregations_df(df, aggregations)
        df_aggregations_with_growth = self.__return_df_with_growth(df_aggregations)
        return df_aggregations_with_growth

    # Function that gets FNR data for a several years and puts them in a DataFrame
    def __get_years(self, from_year, to_year):
        print('Reading data from {}'.format(DATA))

        # Store years of data in list of DataFrames
        df_list = []
        for year in range(from_year, to_year+1):
            df_list.append(self.__get_year(year))

        # Concatenate (stack) list to single DataFrame
        df = pd.concat(df_list)

        # Set all variable names and object type variables to lowecase
        df.columns = [x.lower() for x in df.columns]
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.lower()

        return df

    # Function that gets FNR data for a single year and puts it in a DataFrame
    @staticmethod
    def __get_year(year):
        print('Loading {}'.format(str(year)))
        path = DATA
        df_list = []
        for var in VARS:
            df_list.append(pd.read_sas(''.join([path, '_'.join(['fylke', var, str(year)]), '.sas7bdat']), encoding='iso-8859-1'))
        df = pd.concat(df_list)
        df = df.assign(**{'årgang': pd.Period(value=year, freq='A')})
        return df

    # Funciton that returns a DataFrame where regions are filled in according to region reform of 2018 and 2020
    @staticmethod
    def __fill_missing_regions(df):
        df_fill = df.copy(deep=True)
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

    # Function that returns a DataFrame with MultiIndex
    @staticmethod
    def __set_aggregations_index(df, aggregations):
        df_index = df.copy(deep=True)
        print('Generating aggregations')
        for aggregation in aggregations.get('lists'):
            df_index = df_index.assign(**{aggregation: df['naering'].map(fnr_class.__make_aggregation_mapping(aggregation))})
        for aggregation in aggregations.get('mappings').keys():
            df_index = df_index.assign(**{aggregation: df['naering'].map({x.lower(): aggregation.lower() for x in aggregations.get('mappings')[aggregation]})})
        return df_index.set_index(['årgang', 'nr_variabler', *aggregations.get('lists'), *aggregations.get('mappings'), 'naering'])

    # Function that generates mapping from NR-næring to aggregation
    @staticmethod
    def __make_aggregation_mapping(aggregation):
        path = CATALOGUES
        df = pd.read_sas(''.join([path, 'naering.sas7bdat']), encoding='iso-8859-1')
        return dict(zip(df['naering'].str.lower(), df[aggregation].str.lower()))

    # Function that returns DataFrame for every aggregate in aggregations
    def __make_aggregations_df(self, df, aggregations):
        print('Generating aggregated DataFrame')
        df_list = []
        for aggregation in ['naering',
                            *aggregations.get('lists'),
                            *aggregations.get('mappings').keys()]:
            df_list.append(
                            self
                            .__return_aggregation_df(df, aggregation)
                            .reset_index()
                            .rename(columns={aggregation: 'aggregat'})
                            .set_index(['årgang', 'nr_variabler',
                                        'aggregat', 'aggregering'])
                          )
        return pd.concat(df_list)

    # Function that returns DataFrame aggregated according to aggregation
    def __return_aggregation_df(self, df, aggregation):
        df_aggregation = df.groupby(['årgang', 'nr_variabler', aggregation], dropna=False).sum(min_count=1)
        df_aggregation['aggregering'] = aggregation.lower()
        df_aggregation = df_aggregation[df_aggregation.index.get_level_values(aggregation).isnull() == False]
        return df_aggregation

    # Function that returns DataFrame with volume growth rates
    def __return_df_with_growth(self, df):
        print('Generating volume growth')
        df_vr = df[df.index.get_level_values('nr_variabler') == 'bnp'].droplevel('nr_variabler').sort_index()
        df_fp = df[df.index.get_level_values('nr_variabler') == 'bnpf'].droplevel('nr_variabler').sort_index()
        df_vl = (
            100*(df_fp-df_vr.groupby(['aggregat', 'aggregering']).shift(1))
            .divide(df_vr.groupby(['aggregat', 'aggregering']).shift(1))
            .assign(**{'nr_variabler': 'vlp'})
            .reset_index().set_index(['årgang', 'nr_variabler', 'aggregat', 'aggregering'])
        )
        return pd.concat([df, df_vl.reset_index().set_index(['årgang', 'nr_variabler', 'aggregat', 'aggregering'])])

    # Function that returns a style opbject with selecte variables
    def return_selection(self, years: list, variables: list, aggregation: str, aggregates: list, regions: list, **kwargs) -> pd.DataFrame.style:
        """
        Method that returns a style object for selected years, variables, aggregation, aggregate and regions.
        Optional arguments (**kwargs) are:
        * wide_by: transpose to wide by selected variable
        * sort_by: sort by selected variable or variables
        * round_to: round values to chosen number of decimals

        Example of use:
        * fnr.return_selection([2019,2020], ['bnp'], 'pubagg', ['2x35', '2x41_43'], ['f30', 'f03', 'f34'], wide_by='fylke', round_to=2)
        * fnr.return_selection([2018,2019,2020], ['vlp'], 'naering', ['23720'], ['f30', 'f03', 'f34'], sort_by=['årgang', 'nr_variabler'])
        """
        condition = (
            (self.df.index.get_level_values('årgang').year.isin(years)) &
            (self.df.index.get_level_values('nr_variabler').isin([x.lower() for x in variables])) &
            (self.df.index.get_level_values('aggregering') == aggregation.lower()) &
            (self.df.index.get_level_values('aggregat').isin([x.lower() for x in aggregates]))
        )

        # Reshape DataFrame to long
        df = (
            self.df[condition][regions]
            .reset_index()
            .drop(columns=['aggregering'])
            .melt(id_vars=['årgang', 'nr_variabler', 'aggregat'],
                  value_vars=regions, var_name='fylke', value_name='verdi')
            .set_index(['årgang', 'nr_variabler', 'aggregat', 'fylke'])
        )

        # Reshape DataFrame to wide by chosen variable, if any
        if 'wide_by' in kwargs.keys():
            wide_by = kwargs.get('wide_by')
            df = (
                df
                .reset_index()
                .pivot(index=[x for x in ['årgang', 'nr_variabler', 'aggregat', 'fylke'] if x != wide_by],
                       columns=[wide_by], values='verdi')
            )

        # Sort data by chosen variable or list of variable
        if 'sort_by' in kwargs.keys():
            try:
                df = df.sort_values(kwargs.get('sort_by'))
            except KeyError:
                raise KeyError('Cannot sort by {}.'
                               'Check that variables exist and are not in wide_by'
                               .format(kwargs.get('sort_by')))

        # Return rounded results as style object
        if 'round_to' in kwargs.keys():
            return df.style.format(precision=kwargs.get('round_to'))
        else:
            return df.style

    def to_statbank():
        pass