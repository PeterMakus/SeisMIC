Search.setIndex({docnames:["index","modules/API","modules/corrdb","modules/corrdb/corrdb","modules/correlate","modules/correlate/correlator","modules/correlate/get_started","modules/correlate/stream","modules/get_started","modules/intro","modules/monitor","modules/monitor/dv","modules/monitor/monitor","modules/trace_data","modules/trace_data/waveform"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","modules/API.rst","modules/corrdb.rst","modules/corrdb/corrdb.rst","modules/correlate.rst","modules/correlate/correlator.rst","modules/correlate/get_started.rst","modules/correlate/stream.rst","modules/get_started.rst","modules/intro.rst","modules/monitor.rst","modules/monitor/dv.rst","modules/monitor/monitor.rst","modules/trace_data.rst","modules/trace_data/waveform.rst"],objects:{"seismic.correlate":[[1,0,0,"-","correlate"],[1,0,0,"-","preprocessing_fd"],[1,0,0,"-","preprocessing_stream"],[1,0,0,"-","preprocessing_td"],[1,0,0,"-","stats"],[1,0,0,"-","stream"]],"seismic.correlate.correlate":[[1,1,1,"","Correlator"],[1,3,1,"","calc_cross_combis"],[1,3,1,"","compute_network_station_combinations"],[1,3,1,"","generate_corr_inc"],[1,3,1,"","preprocess_stream"],[1,3,1,"","sort_comb_name_alphabetically"],[1,3,1,"","st_to_np_array"]],"seismic.correlate.correlate.Correlator":[[1,2,1,"","find_existing_times"],[1,2,1,"","find_interstat_dist"],[1,2,1,"","pxcorr"]],"seismic.correlate.preprocessing_fd":[[1,3,1,"","FDfilter"],[1,3,1,"","FDsignBitNormalization"],[1,3,1,"","spectralWhitening"]],"seismic.correlate.preprocessing_stream":[[1,3,1,"","cos_taper"],[1,3,1,"","cos_taper_st"],[1,3,1,"","detrend_st"],[1,3,1,"","stream_filter"]],"seismic.correlate.preprocessing_td":[[1,3,1,"","TDfilter"],[1,3,1,"","TDnormalization"],[1,3,1,"","clip"],[1,3,1,"","detrend"],[1,3,1,"","mute"],[1,3,1,"","normalizeStandardDeviation"],[1,3,1,"","signBitNormalization"],[1,3,1,"","taper"],[1,3,1,"","zeroPadding"]],"seismic.correlate.stats":[[1,1,1,"","CorrStats"]],"seismic.correlate.stats.CorrStats":[[1,4,1,"","defaults"],[1,4,1,"","readonly"]],"seismic.correlate.stream":[[1,1,1,"","CorrBulk"],[1,1,1,"","CorrStream"],[1,1,1,"","CorrTrace"],[1,3,1,"","alphabetical_correlation"],[1,3,1,"","combine_stats"],[1,3,1,"","compare_tr_id"],[1,3,1,"","convert_statlist_to_bulk_stats"],[1,3,1,"","read_corr_bulk"],[1,3,1,"","stack_st"],[1,3,1,"","stack_st_by_group"]],"seismic.correlate.stream.CorrBulk":[[1,2,1,"","copy"],[1,2,1,"","correct_decay"],[1,2,1,"","correct_stretch"],[1,2,1,"","create_corr_stream"],[1,2,1,"","envelope"],[1,2,1,"","extract_multi_trace"],[1,2,1,"","extract_trace"],[1,2,1,"","filter"],[1,2,1,"","mirror"],[1,2,1,"","normalize"],[1,2,1,"","resample"],[1,2,1,"","resample_time_axis"],[1,2,1,"","save"],[1,2,1,"","slice"],[1,2,1,"","smooth"],[1,2,1,"","stretch"],[1,2,1,"","taper"],[1,2,1,"","taper_center"],[1,2,1,"","trim"],[1,2,1,"","wfc"]],"seismic.correlate.stream.CorrStream":[[1,2,1,"","create_corr_bulk"],[1,2,1,"","plot"],[1,2,1,"","select_corr_time"],[1,2,1,"","slide"],[1,2,1,"","stack"]],"seismic.correlate.stream.CorrTrace":[[1,2,1,"","plot"],[1,2,1,"","times"]],"seismic.db":[[1,0,0,"-","corr_hdf5"]],"seismic.db.corr_hdf5":[[1,1,1,"","CorrelationDataBase"],[1,1,1,"","DBHandler"],[1,3,1,"","all_traces_recursive"],[1,3,1,"","co_to_hdf5"],[1,3,1,"","convert_header_to_hdf5"],[1,3,1,"","read_hdf5_header"]],"seismic.db.corr_hdf5.DBHandler":[[1,2,1,"","add_corr_options"],[1,2,1,"","add_correlation"],[1,2,1,"","get_available_channels"],[1,2,1,"","get_available_starttimes"],[1,2,1,"","get_corr_options"],[1,2,1,"","get_data"]],"seismic.monitor":[[1,0,0,"-","dv"],[1,0,0,"-","monitor"],[1,0,0,"-","post_corr_process"],[1,0,0,"-","stretch_mod"]],"seismic.monitor.dv":[[1,1,1,"","DV"],[1,3,1,"","read_dv"]],"seismic.monitor.dv.DV":[[1,2,1,"","plot"],[1,2,1,"","save"],[1,2,1,"","smooth_sim_mat"]],"seismic.monitor.monitor":[[1,1,1,"","Monitor"],[1,3,1,"","average_components"],[1,3,1,"","average_components_wfc"],[1,3,1,"","corr_find_filter"],[1,3,1,"","make_time_list"]],"seismic.monitor.monitor.Monitor":[[1,2,1,"","compute_components_average"],[1,2,1,"","compute_velocity_change"],[1,2,1,"","compute_velocity_change_bulk"],[1,2,1,"","compute_waveform_coherency"],[1,2,1,"","compute_waveform_coherency_bulk"]],"seismic.monitor.post_corr_process":[[1,5,1,"","Error"],[1,5,1,"","InputError"],[1,3,1,"","corr_mat_correct_decay"],[1,3,1,"","corr_mat_decimate"],[1,3,1,"","corr_mat_envelope"],[1,3,1,"","corr_mat_extract_trace"],[1,3,1,"","corr_mat_filter"],[1,3,1,"","corr_mat_mirror"],[1,3,1,"","corr_mat_normalize"],[1,3,1,"","corr_mat_resample"],[1,3,1,"","corr_mat_resample_or_decimate"],[1,3,1,"","corr_mat_resample_time"],[1,3,1,"","corr_mat_shift"],[1,3,1,"","corr_mat_smooth"],[1,3,1,"","corr_mat_stretch"],[1,3,1,"","corr_mat_taper"],[1,3,1,"","corr_mat_taper_center"],[1,3,1,"","corr_mat_trim"],[1,3,1,"","unicode_to_string"]],"seismic.monitor.stretch_mod":[[1,3,1,"","compute_wfc"],[1,3,1,"","est_shift_from_dt_corr"],[1,3,1,"","estimate_reftr_shifts_from_dt_corr"],[1,3,1,"","multi_ref_vchange"],[1,3,1,"","multi_ref_vchange_and_align"],[1,3,1,"","time_shift_estimate"],[1,3,1,"","time_stretch_apply"],[1,3,1,"","time_stretch_estimate"],[1,3,1,"","time_windows_creation"],[1,3,1,"","velocity_change_estimate"],[1,3,1,"","wfc_multi_reftr"]],"seismic.plot":[[1,0,0,"-","plot_correlation"],[1,0,0,"-","plot_dv"],[1,0,0,"-","plot_multidv"],[1,0,0,"-","plot_utils"]],"seismic.plot.plot_correlation":[[1,3,1,"","heat_plot_corr_start"],[1,3,1,"","plot_cst"],[1,3,1,"","plot_ctr"],[1,3,1,"","sect_plot_corr_start"],[1,3,1,"","sect_plot_dist"]],"seismic.plot.plot_dv":[[1,3,1,"","plot_dv"]],"seismic.plot.plot_multidv":[[1,3,1,"","plot_multiple_dv"]],"seismic.plot.plot_utils":[[1,3,1,"","remove_all"],[1,3,1,"","remove_topright"],[1,3,1,"","set_mpl_params"]],"seismic.trace_data":[[1,0,0,"-","waveform"]],"seismic.trace_data.waveform":[[1,1,1,"","FS_Client"],[1,1,1,"","Store_Client"],[1,3,1,"","get_day_in_folder"],[1,3,1,"","read_from_filesystem"]],"seismic.trace_data.waveform.FS_Client":[[1,2,1,"","get_waveforms"]],"seismic.trace_data.waveform.Store_Client":[[1,2,1,"","download_waveforms_mdl"],[1,2,1,"","get_available_stations"],[1,2,1,"","get_waveforms"],[1,2,1,"","read_inventory"],[1,2,1,"","select_inventory_or_load_remote"]],"seismic.utils":[[1,0,0,"-","fetch_func_from_str"],[1,0,0,"-","io"],[1,0,0,"-","miic_utils"]],"seismic.utils.fetch_func_from_str":[[1,3,1,"","func_from_str"]],"seismic.utils.io":[[1,3,1,"","corrmat_to_corrbulk"],[1,3,1,"","flatten"],[1,3,1,"","flatten_recarray"],[1,3,1,"","mat_to_corrtrace"]],"seismic.utils.miic_utils":[[1,3,1,"","convert_timestamp_to_utcdt"],[1,3,1,"","convert_utc_to_timestamp"],[1,3,1,"","discard_short_traces"],[1,3,1,"","filter_stat_dist"],[1,3,1,"","get_valid_traces"],[1,3,1,"","inv_calc_az_baz_dist"],[1,3,1,"","load_header_from_np_array"],[1,3,1,"","nan_moving_av"],[1,3,1,"","resample_or_decimate"],[1,3,1,"","save_header_to_np_array"],[1,3,1,"","stream_require_dtype"],[1,3,1,"","trace_calc_az_baz_dist"],[1,3,1,"","trim_stream_delta"],[1,3,1,"","trim_trace_delta"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:exception"},terms:{"0":[1,3,6,12],"00":[1,6,12],"005":1,"01":[1,6,12],"01t00":1,"02":[1,6],"03":[1,6,12],"04":1,"05":[1,6,12],"06":[1,6],"07":1,"08":[1,6],"09":1,"0z":1,"1":[1,3,6,12,14],"10":[1,6],"100":[1,6],"1000":[1,6],"1001":12,"1024":1,"10th":1,"11":1,"12":[1,6],"12th":1,"13":1,"14":1,"14th":1,"15":1,"15th":1,"16":1,"16th":1,"17th":1,"18":1,"18th":1,"19":1,"1970":1,"1990":14,"19th":1,"1d":1,"2":[1,6],"20":[1,6,12],"200":1,"2006":[1,12],"2009":1,"2010":1,"2011":1,"2015":[6,12],"2016":[6,12],"2019":9,"2021":1,"20th":1,"21":1,"21st":1,"22":1,"22nd":1,"23":1,"24":1,"24_11":1,"24h":1,"25":[1,6],"250":1,"28":1,"289":3,"29":1,"29th":1,"2d":1,"3":[1,6],"30":1,"300":1,"31":1,"317_gje_t_z":1,"35":1,"36":1,"3600":6,"37":1,"38":1,"3d":1,"3rd":1,"4":[1,6],"40":1,"42":1,"43":1,"44":1,"45":1,"46":1,"47":1,"48":1,"49":1,"5":[1,6,12],"50":1,"52":1,"521":1,"54":1,"56":1,"57":1,"58":1,"59":1,"5th":1,"6":1,"60":[6,12],"61":1,"7":8,"72":1,"75":1,"8":[1,6,8],"86398":6,"86400":[6,12],"8th":1,"9":[1,6],"9th":1,"boolean":[1,6],"byte":1,"case":[0,1,3,9],"class":[0,1,3,4,14],"default":1,"do":[1,3,6,14],"export":1,"final":1,"float":[1,6],"function":[0,1,3,6,7,12,14],"import":[1,3,5,6,11,12,14],"int":1,"long":[1,6],"new":[1,6,7],"public":1,"return":[1,3,6,14],"sch\u00f6nefeld":1,"sch\u00f6nfelder":[1,12],"short":1,"switch":1,"transient":1,"true":[1,6,12],"try":[6,12],"while":[1,8],A:[1,5,6],As:[1,3,5,12],By:1,For:[1,3],If:[1,3,6,8,14],In:[1,3,6,9,14],Is:[1,3],It:[1,5,7,11],Its:[0,1],NO:6,No:1,Not:[1,6,8],Of:3,On:1,One:1,Such:1,That:[1,3],The:[0,1,3,4,6,7,11,12,14],There:[1,14],These:[1,6],To:[1,3,6,12,14],Will:1,_:1,_ax:1,_bulk:1,_check_tim:1,_close:3,_get_tim:14,_header:1,_hl:1,_load_loc:14,_t_:1,ab3:1,ab:1,abl:3,abnorm:1,about:[1,3,6,7],abov:[1,3],absmax:1,absolut:[1,6,12],abssum:1,acaus:1,accept:[1,3,6],access:[1,3,13],accord:1,account:[1,9],acess:1,achiev:6,action:1,activ:[1,8],actual:[1,5,7,12],ad:1,add:[1,3],add_corr_opt:1,add_correl:[1,3],add_var_to_dict:1,addit:[1,5],addition:6,adjust:1,advantag:3,affect:1,afili:1,after:[1,3,5,6,8],aftershock:1,afterward:[1,3],again:[1,12],against:1,aggreg:1,akin:7,al:1,algorithm:1,alia:6,alias:1,align:1,all:[1,3,6,14],all_traces_recurs:1,allcombin:1,allow:[1,3,6,9,14],allsimplecombin:1,almost:[1,9],along:[1,8],alow:1,alphabet:[1,3],alphabetical_correl:1,alreadi:[1,3,14],also:[1,6,7,8,9],alter:1,alternativli:1,alwai:[1,3,6],am:1,ambient:[5,9],amount:[1,3],ampitud:1,amplidu:1,amplitud:1,an:[1,2,6,8,9,13],analys:1,angl:1,ani:[1,3,5,6],anoth:1,anti:6,anymor:[1,14],anyth:1,api:0,append:1,applai:1,appli:[0,1],applic:1,approxim:[1,9],april:1,ar:[1,3,6,7,9,12,14],archiv:1,aren:1,arg:[1,6],argument:[0,1,3,4,5,10,11],around:1,arrai:[1,6,7],array_dict:1,array_lik:1,asdf:1,asid:9,aslo:1,aspect:1,assign:7,associ:[1,7],assum:[1,9],assumpt:[9,12],asymmetr:1,attach:1,attach_respons:1,attent:3,attenu:1,attribdict:1,attribut:[1,7],attributeerror:1,author:1,auto:1,autocompon:1,autocorrel:[1,3,9,12],autogen:1,automat:[1,3],avail:[1,2,6,7,12,14],availab:1,avarag:1,averag:1,average_compon:1,average_components_wfc:1,avoid:[1,3,6],avoidwraparound:1,avoidwrapfastlen:1,avoidwrappowertwo:1,awar:1,ax:1,axi:1,az:1,azimhut:1,azimut:1,azimuth:1,b:1,back:1,backscatt:9,band:1,bandpass:[1,6,12],bandstop:1,bartlett:1,base:[1,7,9],base_dir:1,base_directori:1,baseclass:1,baselin:1,bash:5,basi:[1,6],basic:1,batch:6,baz:1,becaus:[1,8],been:[1,3,8],befor:[1,5,6,12],begin:1,behaviour:3,being:1,belong:1,below:[1,6,8,14],best:[1,3],better:6,between:[1,3,6,9,12],betweencompon:1,betweenst:[1,6],bh:1,bhh:1,bhz:1,bia:1,bigger:1,bin:1,binari:[1,11],bit:[1,6],black:1,blackman:1,block:[1,14],booblean:6,bool:1,border:1,both:[1,6,7,14],bottom:[1,12],bound:1,box:3,boxcar:1,br:1,brace:1,brk:1,broadcast:1,bulk:1,butterworth:1,bw:1,by_length:1,bz:1,bzg:6,c:[5,14],cach:1,calc_cross_combi:[1,6],calcul:1,calend:[6,12],calib:1,calibr:1,call:[1,3],callabl:1,calul:1,can:[1,3,5,6,7,8,9,11,12,14],cannot:1,care:[6,7],caus:3,causal:1,cd:[1,8],cdata:1,cdb:[1,3],center:[1,6],center_correl:6,central:1,certain:[1,14],certainli:8,ch0:[1,3,11],ch1:[1,3,11],cha:1,chacomb:3,chaku:12,chan:1,chang:[0,1,6,8,11],channel:[1,3,6,14],charact:1,character:1,cheaper:7,cheby2:1,check:[1,7,14],child:1,choos:1,chosen:[1,3,6],christoph:1,chunk:1,clean:1,client:[1,5,14],clip:[1,6],clipe:6,clock:1,close:[3,6],closer:1,co:[1,3,6],co_to_hdf5:1,code:[1,3,6,8,14],coeffici:1,coher:1,col:1,collect:1,collis:1,color:1,column:[1,6],comb_mseedid:1,combi:1,combin:[1,3,6],combination_method:6,combine_stat:1,come:[1,6,7,8,14],command:8,common:[1,3],compar:[1,12],compare_tr_id:1,comparison:1,compat:[1,8],compens:1,complet:[1,6],complex:1,compon:[1,3,6,12],compress:1,comput:[0,1,3,5,7,8,9,10,11,14],computatio:1,computation:[5,7],compute_components_averag:1,compute_dv:12,compute_network_station_combin:1,compute_velocity_chang:1,compute_velocity_change_bulk:[1,12],compute_waveform_coher:1,compute_waveform_coherency_bulk:1,compute_wfc:1,concept:0,conda:8,conform:1,confus:3,conjuct:5,consequ:9,consid:3,consist:1,constant:[1,6],consult:1,contain:[1,3,14],contamin:1,content:1,context:[1,3],continu:14,conveni:1,convent:[1,3],convert:1,convert_header_to_hdf5:1,convert_statlist_to_bulk_stat:1,convert_timestamp_to_utcdt:1,convert_utc_to_timestamp:1,convolut:1,coord:1,coordin:[1,7,14],copi:1,copyleft:1,copyright:1,cor_inc:1,cor_len:1,core:[1,5,8],corner:1,corr1:1,corr2:1,corr:[1,6],corr_arg:6,corr_data:1,corr_end:[1,3,7],corr_fil:1,corr_find_filt:1,corr_hdf5:[0,3],corr_inc:6,corr_len:[3,6],corr_mat:1,corr_mat_correct_decai:1,corr_mat_decim:1,corr_mat_envelop:1,corr_mat_extract_trac:1,corr_mat_filt:1,corr_mat_mirror:1,corr_mat_norm:1,corr_mat_resampl:1,corr_mat_resample_or_decim:1,corr_mat_resample_tim:1,corr_mat_shift:1,corr_mat_smooth:1,corr_mat_stretch:1,corr_mat_tap:1,corr_mat_taper_cent:1,corr_mat_trim:1,corr_opt:[1,3],corr_start:[1,3,7],corrbulk:[0,1,4],corrdb:3,correct:[1,6],correct_decai:1,correct_stretch:1,correl:[0,7,8,9,12,14],correlationbulk:1,correlationdatabas:[0,1,2],correlationmatrix:1,correlationstream:1,correlationtrac:1,correlaton:1,correltea:3,correspond:1,corrlat:1,corrmat_to_corrbulk:1,corrmatrix:1,corrrel:1,corrstart:1,corrstat:[1,7],corrstream:[0,1,3,4],corrtrac:[0,1,3,4],corrupt:3,cos_tap:1,cos_taper_st:[1,6],cosin:1,cosine_tap:[1,6],costli:1,could:[1,3,5,12,14],count:[1,3],coupl:[3,14],cours:3,cover:1,creat:[1,3,5,6,7,8,11],create_corr_bulk:[1,7],create_corr_stream:1,creation:[1,6],crest:1,critic:6,cross:[1,9,12],crosscompon:1,crossstat:1,cst:[1,3],current:[1,8],curv:1,custom:6,cut:1,d0:6,d:1,dai:[1,6],daili:1,data:[0,1,2,5,6],databas:[1,13],dataloss:3,datapoint:[1,12],dataset:[1,3],datatyp:1,date:[1,6],date_inc:[1,6,12],datetim:1,db:[0,3],dbh:3,dbhandler:[0,1,2],dd:1,de:1,deal:1,debug:[1,6],decai:1,decid:[1,3,6,14],decim:[1,6],deconvolut:1,defin:[1,3,6,11,14],definit:[6,12],delet:[1,6],delete_subdivis:6,delta:1,demean:1,depend:[0,1],depned:1,deriv:1,describ:[1,3,6],descript:1,descriptor:1,desir:1,detail:[1,7],determin:[1,3,6],detrend:[1,6],detrend_st:[1,6],develop:1,deviat:[1,6],di:1,dict:1,dictionai:1,dictionari:[1,3,5,6],dictonari:1,differ:[0,1,3,6,7,12],digit:1,dimens:1,direct:[1,9],direct_output:6,directli:[1,3],directori:[1,6,8,12],dirlist:1,discard:1,discard_short_trac:1,discuss:[3,6],disk:[0,1,6,10],displai:1,disregard:1,dist:1,distanc:[1,7,12],distribut:[8,9],divid:1,doc:1,document:[1,8,12,14],doe:[1,5,7,9],doesn:1,domain:[1,6],don:[1,3,5],done:[1,6],download:[0,1,5,6,13],download_waveforms_mdl:[1,14],downsampl:1,dpi:1,drift:1,driver:1,dt1:1,dt2:1,dt:[1,6,12],dtype:1,due:[1,3],dv:[0,6,10,12],e:[0,1,3,5,6,8,9,11,14],each:[1,3,6,12],earlier:[3,11],earliest:[1,3,6,14],easi:[6,8],easier:3,edg:1,effect:1,egf:[6,12],either:1,elast:0,electromagnet:1,element:1,els:[1,8],embargo:5,emerg:0,emper:6,emploi:9,empti:[1,3],enabl:[5,7],end:[1,12],end_dat:[1,6,12],end_lag:[1,7],end_tim:1,endt:1,endtim:[1,7,14],energi:1,enough:[1,6],ensur:1,env:8,envelop:1,environ:8,equal:1,equival:1,error:[1,3,6],eso:6,essenti:7,est_shift_from_dt_corr:1,estim:[1,6,12,14],estimate_reftr_shifts_from_dt_corr:1,et:1,evalu:1,even:3,ever:3,everi:1,ex_corr:1,exactli:14,exampl:[1,3,6,8,11,14],exce:1,except:1,exclud:1,execut:[1,8],exeed:1,exist:1,exp:1,expect:6,expens:5,experiment:1,explan:1,exponenti:1,extend:1,extend_gap:[1,6],extent:1,extern:1,extra:7,extract:1,extract_multi_trac:1,extract_trac:1,f1:1,f2:1,f3:1,f4:1,f:[1,8],faction:1,factor:[1,6],fail:1,fairli:12,fall:1,fals:[1,6,14],fanci:7,farg:1,fashion:1,fast:1,faster:[1,6],fastest:1,fdfilter:[1,6],fdpreprocecss:6,fdpreprocess:6,fdsignbitnorm:[1,6],fdsn:[1,5,14],februari:1,fed:1,fetch:[1,14],fetch_func_from_str:0,few:12,fft:1,fichtner:9,field:1,fig:12,fig_dir:6,fig_subdir:6,figsiz:1,figur:[1,6],figure_file_nam:1,file:[1,3,5,6,11,12],filenam:1,filenotfounderror:1,filesystem:1,fill:1,filter:[1,6,12],filter_opt:[1,6],filter_stat_dist:1,filtertyp:1,find:[1,11,12],find_existing_tim:1,find_interstat_dist:1,finish:5,fir:1,first:[0,1,3,4,12,14],fit:1,fix:1,flake8:8,flat:1,flatten:1,flatten_recarrai:1,flimit:[1,6],flip:1,float32:1,flush:1,fmt:1,fnmatch:1,folder:[1,6,8,11,12],follow:[1,3,8,14],form:[1,6],format:[1,6,11,14],forward:6,found:6,four:[1,12],fourier:1,fraction:1,frame:1,free:1,freq:1,freq_max:[1,6,12],freq_min:[1,6,12],freqenc:1,freqmax:[1,6],freqmin:[1,6],frequenc:[1,6,12],fridai:1,from:[0,1,3,5,6,7,8,9,10,12,14],fs:1,fs_client:1,fs_persist:1,fs_strategi:1,fs_threshold:1,fsm:1,ftype:[1,6],full:1,fulli:1,fullload:3,func:1,func_from_str:1,funciton:1,funtion:1,further:1,fz:[6,12],g:[1,3,5,6,8,9],gap:1,gener:[1,3,6,14],generate_corr_inc:1,geo:1,geograph:14,geographiclib:8,geometr:1,get:[0,1,2,4,13],get_all_st:1,get_available_channel:[1,3],get_available_st:[1,14],get_available_starttim:[1,3],get_config:1,get_corr_opt:[1,3],get_data:[1,3],get_day_in_fold:1,get_valid_trac:1,get_waveform:[1,14],gfz:1,github:0,given:[1,3,6],gje:1,gji:1,glancec:6,global:1,gnu:1,go:8,gonna:3,good:6,govern:[5,6],greater:1,green:[0,1,6,12,14],ground:12,group:1,grow:14,guarante:1,guid:1,gzip3:1,gzip:1,gzipx:1,h5:[1,3],h5py:[1,3,8],h:[1,6,12],ha:[1,3,7,8],had:3,half:1,halflen:1,ham:1,han:1,handl:[0,1,10,14],handler:1,hann:6,happen:1,hash:1,have:[1,3,5,6,7,8,11,14],hdf5:[1,3,6],hdft5:1,head:8,header1:1,header2:1,header:[1,3,7],heat:1,heat_plot_corr_start:1,heatmap:1,height:1,hejsl:1,help:14,henc:1,here:[1,3,6],hertz:1,hhe:[1,6],hhl:1,hhn:[1,6],hhz:[1,6],high:1,higher:[1,6],highest:1,highpass:1,hilb:1,hilbert:1,hold:[1,3],home:12,homogen:[1,9,12],host:1,how:[1,6,8],howev:[1,7,9],hpc:5,hrv:[1,3,14],html:1,http:1,hz:[1,6],i:[0,1,3,5,6,8,11,14],id:1,idea:6,ideal:1,ident:1,identifi:[1,3],imag:1,implement:3,impli:1,includ:1,include_parti:1,include_partially_select:1,inclus:1,increas:1,increment:[1,6,12],independ:6,index:[0,1],indic:1,indir:1,individu:[1,6],inf:1,influenc:[6,12],info:6,inform:[1,3,6,7,14],inheret:1,inherit:7,iniat:5,initi:[1,5],initialis:[1,5,6,14],inplac:1,input:[1,5,6,14],inputerror:1,insert:1,insid:1,instal:0,instead:[1,7,14],instruct:8,instrument:[1,6],integ:[1,6],intend:1,inter:12,interact:1,intercompon:[1,3,9],interferometr:0,interferometri:[0,1],interpol:1,interpret:1,interst:[1,7,9],interv:[1,6],intimid:6,intorstr:1,introduc:1,introduct:0,inv1:1,inv2:1,inv:1,inv_calc_az_baz_dist:1,inventori:[1,14],invers:1,involv:1,io:0,ipynb:8,irregular:1,item:1,iter:1,its:[1,7],iu:[1,3,14],j:1,joint_norm:[1,6],jointli:1,juici:6,juldai:14,juli:1,june:1,jupyt:8,just:[1,3,6,7,8,14],justifi:9,kbg:6,keep:1,kei:[1,3,6,12],kept:6,keyword:[1,6],kir:6,km:1,know:14,known:[1,3],kwarg:1,l:1,label:1,lag:[1,6,7],laps:[1,6,12],larg:3,last:[1,8],later:[1,6,8],latest:[1,3,6,8,14],latter:1,lead:3,leak:1,least:1,left:1,legal:1,legend:1,len:1,lenght:1,length:[1,3,6,12],lengthtosav:6,lenth:1,less:[1,6],lesser:1,let:6,level:[1,6],librari:1,libver:1,licens:1,life:3,like:[1,3,5,7,8,11,12,14],limit:1,line:1,linewidth:1,link:1,linux:1,list:[1,3,6,7,14],littl:[1,3,6],load:[0,1,3,11,14],load_header_from_np_arrai:1,loader:3,loadmat:1,loc0:3,loc1:3,loc:1,local:[1,14],locat:[1,3,9,14],log:6,log_dir:6,log_level:6,log_subdir:6,logic:3,longer:[1,6],longest:1,look:[5,6,11,12],loop:1,lost:6,lot:1,low:1,lower:[1,12],lowest:6,lowpass:1,lowpasscheby2:1,lowpassfir:1,lru:1,m58a:3,m:[1,6,12],machin:1,mai:[1,14],main:0,make:[1,3,6,7],make_time_list:1,maku:1,manag:[1,3,14],manner:6,manual:8,manz:1,march:1,mark_tim:1,mask:1,mass:14,mat:1,mat_to_corrtrac:1,match:1,matplotlib:[1,8],matric:1,matrici:1,matrix:[1,6],max:1,maxim:1,maxima:1,maximum:[1,6,12],maxixmum:1,maxlat:[1,14],maxlon:[1,14],mb:1,mean:1,meant:3,measur:[1,6,12],median:1,medium:[0,1],memori:[1,6],merg:1,mermaid:8,meta:1,metadata:1,method:[1,3,5,6,7,9,12,14],middl:6,midnight:1,might:[1,3,6,14],miic:[0,1],miic_util:0,mind:1,minim:1,minimax:1,minimum:12,minise:1,minlat:[1,14],minlon:[1,14],mirror:1,miss:[1,14],mitig:1,mix:[1,3],mm:1,mode:[1,3],model:1,modif:1,modifi:[1,3],modul:[0,1,6,8,9,13],mondai:1,monitor:[0,11,12],more:[1,6],most:[1,3,5,6,7,8,9,11],move:1,mpi4pi:8,mpi:[1,5,12],mpio:1,mpirun:[5,12],mseed:[1,6,14],msg:1,much:[1,6],multi:1,multi_ref_panel:1,multi_ref_vchang:1,multi_ref_vchange_and_align:1,multipl:1,multipli:1,must:[1,6],mute:[1,6],my:[3,5,11],my_correl:1,my_sensible_tag:3,mycorrel:5,myfdsn:5,myfil:[1,3],mytag:1,n:[1,5,12],name:1,nan:[1,12],nan_moving_av:1,nan_to_num:1,nativ:[1,6],ndarrai:1,ne:6,necessari:[1,8],need:[1,3,6,14],neg:1,neither:[1,6],nep06:1,nep07:1,nest:1,net0:[1,3,11],net1:[1,3,11],net2:1,net:[1,6],netcomb:3,netlist:1,network1:1,network2:1,network:[1,3,4,14],never:9,next:1,nextfastlen:1,nois:[0,1,4,5,14],non:1,none:[1,3,5],nor:[1,6],norm:1,norm_mean:1,normal:1,normalis:[1,6],normalize_correl:6,normalize_simmat:1,normalizestandarddevi:1,normtyp:1,notabl:7,note:[1,3,6,12],notebook:8,notimplementederror:1,nov:1,novemb:1,now:[6,14],np:1,npt:1,npy:1,npz:[1,11],number:[1,5,6,12],number_cor:5,number_of_cor:12,numer:1,numpi:[1,7,8],o:1,object:[0,1,2,4,6,7,10,12,14],obsolet:6,obspi:[1,5,6,7,8,14],obtain:[1,2],obtaind:1,obviou:[6,12],occur:1,octob:1,offer:[1,3],offset:1,often:1,old:1,onc:[1,3,7,12,14],one:[1,3,6,7,11],ones:1,onli:[1,3,5,6,9,11,12,14],only_mean:1,open:[1,3,8],openmpi:5,oper:1,optim:1,option:[1,5,6],order:[1,6],org:1,orient:1,orientedalong:1,origin:[1,6],other:[1,3,6,12],otherwis:1,our:[3,5,6,12],out:[1,6,7],out_dict:1,outfil:1,output:1,outputdir:1,outputfil:1,outsid:1,over:[1,2,6],overhead:3,overlap:1,overview:[2,13],own:7,p:[1,6],packag:[1,6,8],pad:1,page:[0,1],pai:3,pair:1,pairwis:1,parallel:1,param:[0,1,3,5,6,10],paramet:[1,2,4,12,14],params_exampl:6,paramt:5,parent:7,part:[1,6,14],partial:1,particular:[1,3],particularli:1,pass:[1,6],path:[1,3,5,6,11,14],path_to_fil:12,pathtothisrepo:8,pattern:1,penal:1,per:[1,3,6],percentag:1,percentil:1,perfom:1,perform:1,period:1,permit:1,persist:1,peter:1,phase:1,phd:12,pip:8,pixel:1,place:1,plai:1,plan:8,pleas:1,plot:[0,6,7,10,12],plot_correl:0,plot_cst:1,plot_ctr:1,plot_dv:0,plot_median:1,plot_multidv:0,plot_multiple_dv:1,plot_util:0,plot_vel_chang:[6,12],plt:1,plt_ab:1,plu:1,pm:[1,12],point:[1,6],polici:1,posit:1,possibl:[1,3,6,8,12],post:7,post_corr_process:0,postprocess:[1,3,7],potenti:3,potsdam:1,practic:9,pram:1,pre:9,preced:5,precis:1,preempt:1,preemption:1,prefix:1,preform:1,preinstal:8,prepar:1,preprocecss:6,preprocess:[1,9],preprocess_stream:1,preprocessing_fd:[0,6],preprocessing_stream:[0,6],preprocessing_td:[0,6],preprocessor:1,presenc:[6,9],present:1,prevent:3,previou:11,previous:1,prime:1,principl:8,print:[1,3,5],prior:1,probabl:[3,6],problem:1,proce:1,procedur:12,proces:1,process:[1,3,5,6,7,9],produc:[1,3,7],proj_dir:[5,6],project:[4,5,14],prolong:6,propag:1,properli:1,prov:8,provid:[0,1,3,6,8],pseudo:1,pxcorr:[1,5],py:[5,8,12],pypi:0,pyplot:1,pytest:8,python2:1,python3:8,python:[1,5,6,12,14],pyyaml:8,queri:1,quit:[1,6],r:[1,3],rais:[1,3],ram:6,rang:[1,6,12],rate:[1,6],rather:[1,6],raw:[1,14],rcombi:1,rdcc_nbyte:1,rdcc_nslot:1,rdcc_w0:1,re:1,read:[0,1,2,6,10,13],read_corr_bulk:1,read_dv:[1,11],read_end:6,read_from_filesystem:1,read_hdf5_head:1,read_inc:6,read_inventori:[1,14],read_len:[1,6],read_onli:[1,14],read_start:6,readi:[1,6,11],readonli:1,realis:14,realli:[3,6],reason:[3,9],recalcul:1,recarrai:1,recent:1,recombin:[1,6],recombine_corr_data:1,recombine_subdivis:6,recommend:[1,8,14],record:[3,6,9],recurs:1,reduc:1,ref_tr:1,ref_trc:1,refcorr:1,refer:[1,6],reflect:1,reftr_:1,regard:1,regard_loc:1,rel:1,reli:[3,7],remez:1,remezfir:1,remot:[1,14],remov:[1,6],remove_al:1,remove_nan:1,remove_respons:[1,6],remove_topright:1,replac:[1,3],repo:8,report:1,repositori:[6,8],repres:1,request:1,requir:[1,5,6,14],resampl:1,resample_or_decim:1,resample_time_axi:1,resid:6,respect:[1,6,7],respons:[1,6,14],rest:6,restructur:1,result:[1,6,11,12],retain:1,retriev:[0,1,3,6],return_sim_mat:1,right:[1,12],root:[1,5,6,14],ros3:1,rotat:[1,6],routin:[0,1],row:1,rt:6,rtd:8,rtype:1,rule:1,run:[1,8],s:[0,1,3,6,8,12,14],sac:1,safe:1,said:1,same:[1,5,6,8],sampel:1,sampl:[1,6,12],sampling_r:[1,6],sampling_rate_new:1,save:[0,1,3,6,7,11,12],save_dir:1,save_header_to_np_arrai:1,savez:1,sc:[5,14],scalar:1,scale:1,scalingfactor:1,scipi:[1,8],script:[5,12],sd:[1,14],sds_archiv:1,sdsdir:1,search:[0,1],sec2:1,second:[1,3,5,6,12],second_axi:1,sect_plot_corr_start:1,sect_plot_dist:1,section:[1,7,12,14],see:[1,3,6,12,14],seed:[1,3],seedid:1,segment:1,seiscom:14,seiscomp:1,seismic:[3,5,6,9,11,12,14],seismolog:0,seismomet:9,select:[1,7],select_corr_tim:[1,7],select_inventory_or_load_remot:[1,14],self:1,sen:[1,12],sens:1,sensibl:3,separ:1,seper:1,sequenc:[1,6],seri:[1,6,12],server:14,set:[1,3,4,5,12],set_eida_token:5,set_mpl_param:1,setup:8,sever:[1,3,5,6,8,9,14],shape:1,shift:1,shift_rang:1,shift_step:1,shorter:1,should:[1,3,6,8,12],show:14,shown:[3,6,12],side:[1,6],sign:1,signal:1,signbitnorm:1,sim_mat:1,sim_mat_clim:1,similar:[1,3,12],similarii:1,similarity_percentil:1,simmat:1,simpl:[1,6],simpli:[1,6],singl:1,single_ref:1,single_sid:1,size:[1,14],slice:1,slide:[1,7],slightli:1,slope_frac:1,slot:1,slower:1,smaller:1,smallest:1,smooth:1,smooth_sim_mat:1,so:[1,3,6],sobmodul:6,softwar:0,solut:1,some:[0,1,3,7,12,14],someth:12,sooth:1,sort:[1,7],sort_bi:1,sort_comb_name_alphabet:1,sort_comb_name_aphabet:1,sourc:[1,8,9],space:[1,9],span:1,spatial:12,special:1,specif:[1,4],specifi:1,specify:1,spectal:1,spectra:1,spectralwhiten:[1,6],spectrum:1,speed:1,sphinx:8,sphinxcontrib:8,split:1,spread:1,squar:1,st:[1,5],st_to_np_arrai:1,sta0:1,sta:1,stabil:1,stack:[1,3,6,7,12],stack_34798:3,stack_86398:1,stack_:[1,3],stack_len:1,stack_st:1,stack_st_by_group:1,stacklen:[1,3],standard:[1,3,6],start:[0,1,4,5,7,10],start_dat:[1,6,12],start_lag:[1,7],start_tim:1,startin:12,starting_list:1,startt:1,starttim:[1,3,14],stat0:[1,3,11],stat1:[1,3,11],stat2:1,stat:0,statcomb:3,statfilt:1,station1:1,station2:1,station:[1,3,6,7,9,12,14],stationwid:1,statlist:1,stats1:1,stats2:1,std_factor:[1,6],stdio:1,steinmann:1,step:[1,3,5,6,11],stick:3,still:1,stla:1,stlo:1,storag:[1,6,12],store:[1,3,6,11],store_cli:[1,5,6,14],str:1,straight:6,strategi:1,stream:[0,3,6],stream_copi:1,stream_filt:[1,6],stream_require_dtyp:1,strem:1,stretch:[1,6,12],stretch_mod:0,stretch_rang:[1,6,12],stretch_step:[1,6,12],stretched_mat:1,strftime:1,strictli:1,string:[1,6],stringf:1,strongest:6,strongli:1,strrefmat:1,structur:[1,3,6,14],strvec:1,studi:9,stuff:6,style:1,subdir:[6,12],subdirectori:6,subdivid:1,subdivis:[1,3,6],subfold:[6,12],sublist:1,submodul:1,subsequ:[1,6,12],subset:1,substanti:6,subsurfac:1,success:1,suffieci:3,summar:1,support:1,suppos:1,sure:[1,6],surpress:1,swmr:1,syntax:[1,6],system:[1,8],t:[1,3,5],t_width:1,ta:3,tag:[1,2],take:[1,5,6,7,11],taken:1,talk:1,taper:[1,6],taper_arg:1,taper_at_mask:[1,6],taper_cent:1,taper_len:[1,6],target:1,tdfilter:[1,6],tdnormal:1,tdpreprocecss:6,tdpreprocess:6,team:1,techniqu:[9,12],tempor:[0,1],tend:1,test:[1,8,12],than:[1,6],thei:[1,3,6,7],them:[1,6,14],theme:8,themselv:1,theori:9,thi:[1,3,5,6,7,8,9,11,12,14],thio:1,those:[1,3,6,11],thought:1,thre:1,three:[1,6],threhold:1,threshold:[1,6],through:1,throught:8,thu:1,thumb:1,thursdai:1,tick:1,time:[1,5,6,7,9,12],time_shift:1,time_shift_estim:1,time_stretch_appli:1,time_stretch_estim:1,time_vect:1,time_window:1,time_windows_cr:1,timelimit:1,timestamp:1,titl:1,tlim:1,todai:9,togeth:1,tomographi:9,too:1,tool:1,top:1,total:1,tqdm:8,tr0:1,tr1:1,tr2:1,tr:1,trace:[0,1,6,7],trace_calc_az_baz_dist:1,trace_data:[0,5,14],traceback:1,tracec:1,track:1,track_ord:1,tranform:1,transform:1,translat:1,treat:1,trend:1,trim:1,trim_stream_delta:1,trim_trace_delta:1,truncat:1,tsai:9,tuesdai:1,tupl:1,tutori:0,tw:[1,12],tw_len:[1,6,12],tw_mat:1,tw_start:[1,6,12],twice:1,two:[1,3,5,9,14],txt:5,type:[1,3,6,7],typeerror:1,under:1,understand:6,unexpect:3,unicod:1,unicode_to_str:1,unit:1,unknown:1,unkown:1,unstack:[1,3],unus:6,up:1,updat:1,upon:[1,12],upper:[1,12],upsampl:1,us:[0,1,3,5,6,7,8,11,12,14],usag:[1,6,12],usecas:3,user:[1,3,6],userblock_s:1,usual:[1,6],utc:1,utcdatetim:[1,14],utcdatetimeorlist:1,utcdt:1,utf:1,util:0,v108:1,v110:1,v112:1,v:1,vale_typ:1,valid:1,valu:[1,6],value_typ:1,valueerror:1,vari:[1,9],varianc:1,variat:1,variou:[1,9],varying_loc:1,vector:1,vel_chang:[6,12],veloc:[0,1,6,11],velocity_change_estim:1,veri:[1,5,12],version:[1,8],vertic:1,vertor:1,vfd:1,via:[0,1],vicin:[9,12],view:[1,6],vital:7,w:[1,3],wa:1,wai:[1,6,14],want:[1,3,5,6,8,11,12,14],warn:[1,6],wave:[0,9],waveform:[0,3,5,6,7,13],waveform_coher:1,we:[1,3,5,6,8,12,14],weaver:1,wednesdai:1,wegler:[1,12],weight:1,weigth:1,well:[1,14],were:[1,7,11],wfc:1,wfc_multi_reftr:1,what:6,when:[1,3,6,14],where:[1,3,5,6],wherea:[1,3],whether:[1,6,14],which:[1,3,5,6,7,9,12,14],whiten:1,whole:1,whose:1,wide:4,width:1,wiggl:1,wildcard:[1,3,6],win_half_len:1,win_inc:1,win_len:[1,6,12],window:[1,6,7,12],window_length:1,windowlength:1,wish:1,within:1,without:1,withouta:1,won:1,work:[0,1,3,6],would:[1,6,12,14],wrap:1,write:[0,1,2,6],written:[1,6,14],wrong:1,wsize:1,wtype:1,www:1,x:[1,5],xlim:1,xlimit:1,xml:14,xn:1,xtick:1,y:[1,6,12],yaml:[0,1,3,5,6,10,11],yaml_f:12,year:[1,14],yet:[6,8],yield:1,ylim:1,ylimit:1,yml:[5,8],you:[1,3,5,6,8,11,12,14],your:[0,1,3,4,5,8,12,13],yp:1,ytick:1,yyyi:1,z:[1,3],zero:[1,6],zeropad:1,zne:[1,6],zrt:[1,6]},titles:["SeisMIC library documentation","<code class=\"docutils literal notranslate\"><span class=\"pre\">SeisMIC</span></code> API","Save and Load Correlations","CorrelationDataBase and DBHandler objects","Compute and Handle Correlations","The Correlator Object","Get Started (Compute your first Noise Correlations)","CorrTrace, CorrStream, and CorrBulk classes","Getting Started with SeisMIC","Introduction: Noise Interferometry with Green\u2019s Function Retrieval","Monitor Velocity Changes","Reading and Handling of DV objects","Compute Velocity Changes","Trace data","Module to Download and Read Waveform Data"],titleterms:{"class":7,"function":9,The:5,access:14,actual:6,an:[3,14],api:1,argument:[6,12],avail:3,chang:[10,12],comput:[4,6,12],content:[0,13],corr_hdf5:1,corrbulk:7,correl:[1,2,3,4,5,6],correlationdatabas:3,corrstream:7,corrtrac:7,data:[3,13,14],databas:14,db:1,dbhandler:3,depend:8,disk:11,document:0,download:[8,14],dv:[1,11],fetch_func_from_str:1,first:6,flowchart:0,from:11,get:[3,6,8,14],github:8,green:9,handl:[4,11],indic:0,instal:8,interact:0,interferometri:9,introduct:9,io:1,librari:0,load:2,miic_util:1,modul:14,monitor:[1,10],mpi:8,network:6,nois:[6,9],object:[3,5,11],obtain:3,over:3,overview:[3,14],param:12,paramet:[3,6],plot:[1,11],plot_correl:1,plot_dv:1,plot_multidv:1,plot_util:1,post_corr_process:1,preprocess:6,preprocessing_fd:1,preprocessing_stream:1,preprocessing_td:1,project:6,pypi:8,python:8,read:[3,11,14],retriev:9,routin:13,s:9,save:2,seismic:[0,1,8,13],set:6,specif:6,start:[6,8,12],stat:1,stream:1,stretch_mod:1,tabl:0,tag:3,trace:13,trace_data:1,tutori:8,util:1,veloc:[10,12],via:8,waveform:[1,14],wide:6,work:13,write:3,yaml:12,your:[6,14]}})