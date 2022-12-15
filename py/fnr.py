# Author: Magnus Helliesen
import numpy as np
import pandas as pd

PATH_DATA = '/ssb/stamme02/nasjregn/fnr_mr2022/wk24/'
PATH_CATALOGUES = '/ssb/stamme02/nasjregn/systemkataloger_mr2022/kat/'


class fnr_class:
    """
    Docstring TBA
    """

    def __init__(self, from_year, to_year, aggregations):
        self.__aggregations = aggregations
        # Fill in with empty lists and dicts if not supplied
        if self.__aggregations.get('lists') is None:
            self.__aggregations = {**self.__aggregations, **{'lists': {}}}
        if self.__aggregations.get('mappings') is None:
            self.__aggregations = {**self.__aggregations, **{'mappings': {}}}
        self.__df = self.__setup_class(from_year, to_year, self.__aggregations)

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
        df_list = []
        for year in range(from_year, to_year+1):
            df_list.append(self.__get_year(year))
        df = pd.concat(df_list)
        return df

    # Function that gets FNR data for a single year and puts it in a DataFrame
    @staticmethod
    def __get_year(year):
        path = PATH_DATA
        df_list = []
        for var in ['prod', 'pin', 'bnp', 'bnpf', 'brin', 'lkost', 'syss']:
            df_list.append(pd.read_sas(''.join([path, '_'.join(['fylke', var, str(year)]), '.sas7bdat']), encoding='iso-8859-1'))
        df = pd.concat(df_list)
        df['årgang'] = pd.Period(value=year, freq='A')
        return df

    # Funciton that returns a DataFrame where regions are filled in according to region reform of 2020
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
        for aggregation in aggregations.get('lists'):
            df_index = df_index.assign(**{aggregation: df['naering'].map(fnr_class.__make_aggregation_mapping(aggregation))})
        for aggregation in aggregations.get('mappings').keys():
            df_index = df_index.assign(**{aggregation: df['naering'].map({x: aggregation for x in aggregations.get('mappings')[aggregation]})})
        return df_index.set_index(['årgang', 'nr_variabler', *aggregations.get('lists'), *aggregations.get('mappings'), 'naering'])

    # Function that generates mapping from NR-næring to aggregation
    @staticmethod
    def __make_aggregation_mapping(aggregation):
        path = PATH_CATALOGUES
        df = pd.read_sas(''.join([path, 'naering.sas7bdat']), encoding='iso-8859-1')
        return dict(zip(df['naering'], df[aggregation]))

    # Function that returns dataframe for every aggregate in aggregations
    def __make_aggregations_df(self, df, aggregations):
        df_list = []
        for aggregation in ['naering', *aggregations.get('lists'), *aggregations.get('mappings').keys()]:
            df_list.append(
                            self
                            .__return_aggregation_df(df, aggregation)
                            .reset_index()
                            .rename(columns={aggregation: 'aggregat'})
                            .set_index(['årgang', 'nr_variabler',
                                        'aggregat', 'aggregering'])
                          )
        df_aggregations = pd.concat(df_list)
        return df_aggregations

    # Function that returns dataframe aggregated according to aggregation
    def __return_aggregation_df(self, df, aggregation):
        df_aggregation = df.groupby(['årgang', 'nr_variabler', aggregation], dropna=False).sum(min_count=1)
        df_aggregation['aggregering'] = aggregation.upper()
        df_aggregation = df_aggregation[df_aggregation.index.get_level_values(aggregation).isnull() == False]
        return df_aggregation

    # Function that returns dataframe with volume growth rates
    def __return_df_with_growth(self, df):
        df_vr = df[df.index.get_level_values('nr_variabler') == 'BNP'].droplevel('nr_variabler').sort_index()
        df_fp = df[df.index.get_level_values('nr_variabler') == 'BNPF'].droplevel('nr_variabler').sort_index()
        df_vl = (
            100*(df_fp-df_vr.groupby(['aggregat', 'aggregering']).shift(1))
            .divide(df_vr.groupby(['aggregat', 'aggregering']).shift(1))
            .assign(**{'nr_variabler': 'VLP'})
            .reset_index().set_index(['årgang', 'nr_variabler', 'aggregat', 'aggregering'])
        )
        df = pd.concat([df, df_vl.reset_index().set_index(['årgang', 'nr_variabler', 'aggregat', 'aggregering'])])
        return df

    # Function that returns dataframe ...
    def return_aggregate(self, aggregate, variable):
        condition = ((self.df.index.get_level_values('aggregat') == aggregate) & 
                     (self.df.index.get_level_values('nr_variabler') == variable))
        return self.df[condition].style.format(decimal=',', precision=2)

    def to_statbank():
        pass
