import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, interact_manual, FloatSlider, IntSlider, IntRangeSlider, Select, SelectMultiple, Checkbox, Dropdown, HBox, VBox, interactive_output, Label, Layout, HTML
from IPython.display import display


def plott_fylke(df, fnr):
    aargang_min = df.index.get_level_values('årgang').min().year
    aargang_max = df.index.get_level_values('årgang').max().year
    variabler = [x for x in df.index.get_level_values('nr_variabler').unique().to_list() if x != 'bnpf']
    aggregeringer = df.index.get_level_values('aggregering').unique().to_list()
    regionsreformer = ['-2017', '2018-2019', '2020-']

    def lag_aggregater(variabel, aggregering):
        if variabel == 'brin':
            return [x for x in df[df.index.get_level_values('aggregering') == aggregering].index.get_level_values('aggregat').unique().to_list() if x[0] == '8']
        else:
            return [x for x in df[df.index.get_level_values('aggregering') == aggregering].index.get_level_values('aggregat').unique().to_list() if x[0] == '2']

    def lag_fylker(regionsreform):
        if regionsreform == '-2017':
            return ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f23']
        if regionsreform == '2018-2019':
            return ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f14', 'f15', 'f50', 'f18', 'f19', 'f20', 'f21', 'f23']
        if regionsreform == '2020-':
            return ['f30', 'f03', 'f34', 'f38', 'f42', 'f11', 'f46', 'f15', 'f50', 'f18', 'f54', 'f21', 'f23']

    aargang_wgt = IntRangeSlider(value=[aargang_min, aargang_max], min=aargang_min, max=aargang_max)
    variabel_wgt = Dropdown(options=variabler, description='Variabel', style={'description_width': '3cm'})
    aggregering_wgt = Dropdown(options=aggregeringer, value=aggregeringer[1], description = 'Aggregering', style={'description_width': '3cm'})
    aggregat_wgt = Select(options=lag_aggregater(variabler[0], aggregeringer[1]), value=lag_aggregater(variabler[0], aggregeringer[1])[0], description = 'Aggregat', style={'description_width': '3cm'}, rows=10)
    regionsreformer_wgt = Dropdown(options=regionsreformer, value=regionsreformer[2], description = 'Regionsreform', style={'description_width': '3cm'})
    fylker_wgt = SelectMultiple(options=lag_fylker(regionsreformer[2]), value=lag_fylker(regionsreformer[2]), description = 'Fylke', style={'description_width': '3cm'}, rows=10)
    stack_wgt = Checkbox(value=True, description='Stable stolper')
    first_diff_wgt = Checkbox(value=False, description='Førstedifferanse')

    def oppdater_aggregater(arg):
        aggregat_wgt.options = lag_aggregater(variabel_wgt.value, aggregering_wgt.value)

    def oppdater_fylker(arg):
        fylker_wgt.options = lag_fylker(regionsreformer_wgt.value)

    variabel_wgt.observe(oppdater_aggregater)
    aggregering_wgt.observe(oppdater_aggregater)
    regionsreformer_wgt.observe(oppdater_fylker)

    def skriv_tabell(aargang, aggregering, aggregat, variabel, fylker, stack, forste_diff):
        try:
            (
                fnr
                .return_selection(aggregering, list(range(aargang[0], aargang[1]+1)), [variabel], aggregat, fylker, wide_by='fylke', first_diff=forste_diff)
                .data
                .reset_index().set_index('årgang')
                .plot
                .bar(figsize=(15, 7.5), stacked=stack)
            )
            plt.xlabel('Årgang')
            plt.legend(ncol=10)
            plt.grid(axis='y')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.show()
        except TypeError:
            pass

    ui = HBox([VBox([stack_wgt, first_diff_wgt, aargang_wgt, variabel_wgt]), VBox([aggregering_wgt, aggregat_wgt]), VBox([regionsreformer_wgt, fylker_wgt])])
    out = interactive_output(skriv_tabell, {'aargang': aargang_wgt,
                                            'aggregering': aggregering_wgt,
                                            'aggregat': aggregat_wgt,
                                            'variabel': variabel_wgt,
                                            'fylker': fylker_wgt,
                                            'stack': stack_wgt,
                                            'forste_diff': first_diff_wgt})
    display(ui, out)


def plott_naering(df, agg_dict, fnr):
    aargang_min = df.index.get_level_values('årgang').min().year
    aargang_max = df.index.get_level_values('årgang').max().year
    variabler = [x for x in df.index.get_level_values('nr_variabler').unique().to_list() if x != 'bnpf']
    aggregeringer = [x for x in df.index.get_level_values('aggregering').unique().to_list() if x != 'naering']
    regionsreformer = ['-2017', '2018-2019', '2020-']

    def lag_aggregater(variabel, aggregering):
        if variabel == 'brin':
            return [x for x in df[df.index.get_level_values('aggregering') == aggregering].index.get_level_values('aggregat').unique().to_list() if x[0] == '8']
        else:
            return [x for x in df[df.index.get_level_values('aggregering') == aggregering].index.get_level_values('aggregat').unique().to_list() if x[0] == '2']

    def lag_fylker(regionsreform):
        if regionsreform == '-2017':
            return ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f23']
        if regionsreform == '2018-2019':
            return ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f14', 'f15', 'f50', 'f18', 'f19', 'f20', 'f21', 'f23']
        if regionsreform == '2020-':
            return ['f30', 'f03', 'f34', 'f38', 'f42', 'f11', 'f46', 'f15', 'f50', 'f18', 'f54', 'f21', 'f23']

    aargang_wgt = IntRangeSlider(value=[aargang_min, aargang_max], min=aargang_min, max=aargang_max)
    variabel_wgt = Dropdown(options=variabler, description='Variabel', style={'description_width': '3cm'})
    aggregering_wgt = Dropdown(options=aggregeringer, value=aggregeringer[1], description = 'Aggregering', style={'description_width': '3cm'})
    aggregat_wgt = SelectMultiple(options=lag_aggregater(variabler[0], aggregeringer[1]), value=lag_aggregater(variabler[0], aggregeringer[1]), description = 'Aggregat', style={'description_width': '3cm'}, rows=10)
    regionsreformer_wgt = Dropdown(options=regionsreformer, value=regionsreformer[2], description = 'Regionsreform', style={'description_width': '3cm'})
    fylker_wgt = Select(options=lag_fylker(regionsreformer[2]), value=lag_fylker(regionsreformer[2])[0], description = 'Fylke', style={'description_width': '3cm'}, rows=10)
    stack_wgt = Checkbox(value=True, description='Stable stolper')
    first_diff_wgt = Checkbox(value=False, description='Førstedifferanse')
    det_wgt = Checkbox(value=False, description='Vis næringer')

    def oppdater_aggregater(arg):
        aggregat_wgt.options = lag_aggregater(variabel_wgt.value, aggregering_wgt.value)

    def oppdater_fylker(arg):
        fylker_wgt.options = lag_fylker(regionsreformer_wgt.value)

    variabel_wgt.observe(oppdater_aggregater)
    aggregering_wgt.observe(oppdater_aggregater)
    regionsreformer_wgt.observe(oppdater_fylker)

    def skriv_tabell(aargang, aggregering, aggregat, variabel, fylker, stack, forste_diff, det):
        if det:
            aggregering_ = 'naering'
            aggregat_ = []
            for agg in aggregat:
                aggregat_.extend(agg_dict.get(aggregering).get(agg))
        else:
            aggregering_ = aggregering
            aggregat_ = aggregat
        try:
            (
                fnr
                .return_selection(aggregering_, list(range(aargang[0], aargang[1]+1)), [variabel], aggregat_, fylker, wide_by='aggregat', first_diff=forste_diff)
                .data
                .reset_index().set_index('årgang')
                .plot
                .bar(figsize=(15, 7.5), stacked=stack)
            )
            plt.xlabel('Årgang')
            plt.legend(ncol=10)
            plt.grid(axis='y')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.show()
        except TypeError:
            pass

    ui = HBox([VBox([stack_wgt, first_diff_wgt, det_wgt, aargang_wgt, variabel_wgt]), VBox([aggregering_wgt, aggregat_wgt]), VBox([regionsreformer_wgt, fylker_wgt])])
    out = interactive_output(skriv_tabell, {'aargang': aargang_wgt,
                                            'aggregering': aggregering_wgt,
                                            'aggregat': aggregat_wgt,
                                            'variabel': variabel_wgt,
                                            'fylker': fylker_wgt,
                                            'stack': stack_wgt,
                                            'forste_diff': first_diff_wgt,
                                            'det': det_wgt})
    display(ui, out)


def lag_tabell(df, agg_dict, fnr):
    aargang_min = df.index.get_level_values('årgang').min().year
    aargang_max = df.index.get_level_values('årgang').max().year
    variabler = [x for x in df.index.get_level_values('nr_variabler').unique().to_list() if x != 'bnpf']
    aggregeringer = [x for x in df.index.get_level_values('aggregering').unique().to_list() if x != 'naering']
    regionsreformer = ['-2017', '2018-2019', '2020-']

    def lag_aggregater(variabel, aggregering):
        if variabel == 'brin':
            return [x for x in df[df.index.get_level_values('aggregering') == aggregering].index.get_level_values('aggregat').unique().to_list() if x[0] == '8']
        else:
            return [x for x in df[df.index.get_level_values('aggregering') == aggregering].index.get_level_values('aggregat').unique().to_list() if x[0] == '2']

    def lag_fylker(regionsreform):
        if regionsreform == '-2017':
            return ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f23']
        if regionsreform == '2018-2019':
            return ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f14', 'f15', 'f50', 'f18', 'f19', 'f20', 'f21', 'f23']
        if regionsreform == '2020-':
            return ['f30', 'f03', 'f34', 'f38', 'f42', 'f11', 'f46', 'f15', 'f50', 'f18', 'f54', 'f21', 'f23']

    aargang_wgt = IntRangeSlider(value=[aargang_min, aargang_max], min=aargang_min, max=aargang_max)
    variabel_wgt = Dropdown(options=variabler, description='Variabel', style={'description_width': '3cm'})
    aggregering_wgt = Dropdown(options=aggregeringer, value=aggregeringer[1], description = 'Aggregering', style={'description_width': '3cm'})
    aggregat_wgt = SelectMultiple(options=lag_aggregater(variabler[0], aggregeringer[1]), value=lag_aggregater(variabler[0], aggregeringer[1]), description = 'Aggregat', style={'description_width': '3cm'}, rows=10)
    regionsreformer_wgt = Dropdown(options=regionsreformer, value=regionsreformer[2], description = 'Regionsreform', style={'description_width': '3cm'})
    fylker_wgt = SelectMultiple(options=lag_fylker(regionsreformer[2]), value=lag_fylker(regionsreformer[2]), description = 'Fylke', style={'description_width': '3cm'}, rows=10)
    round_wgt = Checkbox(value=True, description='Rund av')
    first_diff_wgt = Checkbox(value=False, description='Førstedifferanse')
    wide_by_wgt = Dropdown(options=['årgang', 'nr_variabler', 'aggregat', 'fylke'], value='fylke', description = 'Kolonnevariabel', style={'description_width': '3cm'})
    det_wgt = Checkbox(value=False, description='Vis næringer')

    def oppdater_aggregater(arg):
        aggregat_wgt.options = lag_aggregater(variabel_wgt.value, aggregering_wgt.value)

    def oppdater_fylker(arg):
        fylker_wgt.options = lag_fylker(regionsreformer_wgt.value)

    variabel_wgt.observe(oppdater_aggregater)
    aggregering_wgt.observe(oppdater_aggregater)
    regionsreformer_wgt.observe(oppdater_fylker)

    def skriv_tabell(aargang, aggregering, aggregat, variabel, fylker, rund_av, forste_diff, bred, det):
        if det:
            aggregering_ = 'naering'
            aggregat_ = []
            for agg in aggregat:
                aggregat_.extend(agg_dict.get(aggregering).get(agg))
        else:
            aggregering_ = aggregering
            aggregat_ = aggregat
        try:
            display(
                fnr
                .return_selection(aggregering_, list(range(aargang[0], aargang[1]+1)), [variabel], aggregat_, fylker, wide_by=bred, round_to=(0 if rund_av else 1), first_diff=forste_diff)
            )
        except TypeError:
            pass

    ui = HBox([VBox([round_wgt, first_diff_wgt, det_wgt, wide_by_wgt, aargang_wgt, variabel_wgt]), VBox([aggregering_wgt, aggregat_wgt]), VBox([regionsreformer_wgt, fylker_wgt])])
    out = interactive_output(skriv_tabell, {'aargang': aargang_wgt,
                                            'aggregering': aggregering_wgt,
                                            'aggregat': aggregat_wgt,
                                            'variabel': variabel_wgt,
                                            'fylker': fylker_wgt,
                                            'rund_av': round_wgt,
                                            'forste_diff': first_diff_wgt,
                                            'bred': wide_by_wgt,
                                            'det': det_wgt})
    display(ui, out)


def plott_naering2(df, agg_dict, fnr):
    aargang_min = df.index.get_level_values('årgang').min().year
    aargang_max = df.index.get_level_values('årgang').max().year
    variabler = [x for x in df.index.get_level_values('nr_variabler').unique().to_list() if x != 'bnpf']
    aggregeringer = [x for x in df.index.get_level_values('aggregering').unique().to_list() if x != 'naering']
    regionsreformer = ['-2017', '2018-2019', '2020-']

    def lag_aggregater(variabel, aggregering):
        if variabel == 'brin':
            return [x for x in df[df.index.get_level_values('aggregering') == aggregering].index.get_level_values('aggregat').unique().to_list() if x[0] == '8']
        else:
            return [x for x in df[df.index.get_level_values('aggregering') == aggregering].index.get_level_values('aggregat').unique().to_list() if x[0] == '2']

    def lag_fylker(regionsreform):
        if regionsreform == '-2017':
            return ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f23']
        if regionsreform == '2018-2019':
            return ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f14', 'f15', 'f50', 'f18', 'f19', 'f20', 'f21', 'f23']
        if regionsreform == '2020-':
            return ['f30', 'f03', 'f34', 'f38', 'f42', 'f11', 'f46', 'f15', 'f50', 'f18', 'f54', 'f21', 'f23']

    aargang_wgt = IntRangeSlider(value=[aargang_min, aargang_max], min=aargang_min, max=aargang_max)
    variabel_wgt = Dropdown(options=variabler, description='Variabel', style={'description_width': '3cm'})
    aggregering_wgt = Dropdown(options=aggregeringer, value=aggregeringer[1], description = 'Aggregering', style={'description_width': '3cm'})
    aggregat_wgt = SelectMultiple(options=lag_aggregater(variabler[0], aggregeringer[1]), value=lag_aggregater(variabler[0], aggregeringer[1]), description = 'Aggregat', style={'description_width': '3cm'}, rows=10)
    regionsreformer_wgt = Dropdown(options=regionsreformer, value=regionsreformer[2], description = 'Regionsreform', style={'description_width': '3cm'})
    fylker_wgt = Select(options=lag_fylker(regionsreformer[2]), value=lag_fylker(regionsreformer[2])[0], description = 'Fylke', style={'description_width': '3cm'}, rows=10)
    sort_wgt = Checkbox(value=True, description='Sorter')
    first_diff_wgt = Checkbox(value=False, description='Førstedifferanse')
    wide_by_wgt = Dropdown(options=['årgang', 'nr_variabler', 'aggregat', 'fylke'], value='fylke', description = 'Kolonnevariabel', style={'description_width': '3cm'})
    det_wgt = Checkbox(value=False, description='Vis næringer')

    def oppdater_aggregater(arg):
        aggregat_wgt.options = lag_aggregater(variabel_wgt.value, aggregering_wgt.value)

    def oppdater_fylker(arg):
        fylker_wgt.options = lag_fylker(regionsreformer_wgt.value)

    variabel_wgt.observe(oppdater_aggregater)
    aggregering_wgt.observe(oppdater_aggregater)
    regionsreformer_wgt.observe(oppdater_fylker)

    def skriv_tabell(aargang, aggregering, aggregat, variabel, fylker, sort, forste_diff, bred, det):
        if det:
            aggregering_ = 'naering'
            aggregat_ = []
            for agg in aggregat:
                aggregat_.extend(agg_dict.get(aggregering).get(agg))
        else:
            aggregering_ = aggregering
            aggregat_ = aggregat
        try:
            (
                fnr
                .return_selection(aggregering_, list(range(aargang[0], aargang[1]+1)), [variabel], aggregat_, fylker, wide_by='årgang', first_diff=forste_diff, sort_by=[str(aargang[-1])] if sort else [])
                .data
                .reset_index().set_index('aggregat')
                .plot
                .bar(figsize=(15, 7.5))
            )
            plt.xlabel('Aggregat')
            plt.legend(ncol=10)
            plt.grid(axis='y')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.show()
        except TypeError:
            pass

    ui = HBox([VBox([sort_wgt, first_diff_wgt, det_wgt, wide_by_wgt, aargang_wgt, variabel_wgt]), VBox([aggregering_wgt, aggregat_wgt]), VBox([regionsreformer_wgt, fylker_wgt])])
    out = interactive_output(skriv_tabell, {'aargang': aargang_wgt,
                                            'aggregering': aggregering_wgt,
                                            'aggregat': aggregat_wgt,
                                            'variabel': variabel_wgt,
                                            'fylker': fylker_wgt,
                                            'sort': sort_wgt,
                                            'forste_diff': first_diff_wgt,
                                            'bred': wide_by_wgt,
                                            'det': det_wgt})
    display(ui, out)