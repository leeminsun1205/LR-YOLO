export type Family='low_light'|'haze';
export type Model={id:string;label:string;architecture:string;degradation:Family;stage:string;case_id:number;classes:string[];available:boolean};
export type Box={bbox:number[];label:string;score:number;difficult?:boolean};
export type Example={id:string;image_id:string;source:string;split:string;already_degraded:boolean;annotations?:Box[]};
export type Prediction={model_id:string;label:string;boxes:Box[];classes:string[];metadata:{checkpoint_epoch?:number;completed_epochs?:number;run_target_epochs?:number;checkpoint_sha256?:string;provenance_verified:boolean;stage:string;architecture:string;case_id:number};timing:{inference_ms:number;model_load_seconds:number;scope:string};reconstruction_url?:string;annotated_url:string};
export type Job={id:string;status:string;progress:string;error?:string;width:number;height:number;input_url:string;original_url:string;results:Prediction[];degradation:{mode:string;gamma:number;beta:number};elapsed_seconds:number};
export type Report={id:string;label:string;source:string;degradation:Family;architecture:string;dataset:string;metric:string;epochs:number;ap50?:number;train_hours?:number;vram_gb?:number;vram_scope?:string;budget_group?:string};
