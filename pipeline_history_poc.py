
import pandas as pd
import numpy as np
import os
import datetime as datetime
from datetime import date, datetime, timedelta

def run_pipeline_history_poc():
    print("Starting Pipeline History PoC...")

    # 1. Load Stage Lookups
    print("Loading StageLookups.csv...")
    stagelookup = pd.read_csv('StageLookups.csv')
    lookup_dict = dict(zip(stagelookup['StageName'], stagelookup['Lookup']))

    # 2. Load Opportunity Stages (Mocking SOQL: SELECT ... FROM OpportunityStage)
    print("Loading dummy_stages_sobject.csv...")
    stages = pd.read_csv('dummy_stages_sobject.csv')
    stages = stages[['MasterLabel','IsWon','IsClosed']]
    stages = stages.rename(columns={'MasterLabel':'StageName'})

    # 3. Load Field History (Mocking SOQL: SELECT ... FROM OpportunityFieldHistory)
    print("Loading dummy_field_history.csv...")
    fieldHistory = pd.read_csv('dummy_field_history.csv')
    fieldHistory= fieldHistory.sort_values(by=['Id'],ascending=True)
    fieldHistory.drop_duplicates(
        subset=["OpportunityId", "Field", "CreatedDate"], keep="last", inplace=True
    )
    # Convert CreatedDate
    fieldHistory["CreatedDate"] = pd.to_datetime(fieldHistory["CreatedDate"])
    # Note: Skipping timezone conversion for simplicity in PoC or assuming UTC input

    # 4. Load Opportunities (Mocking SOQL: SELECT ... FROM Opportunity)
    print("Loading dummy_opportunities.csv...")
    opps_raw = pd.read_csv('dummy_opportunities.csv')
    opps = opps_raw[['Id', 'ARR_Amount__c', 'StageName', 'CloseDate', 'CreatedDate', 'CurrencyIsoCode', 'NextStep']].copy()
    
    opps.rename(columns={"CreatedDate": "created"}, inplace=True)

    print("Loading currency.csv...")
    currency = pd.read_csv("./currency.csv")

    oppcurrency = opps[["Id", "CurrencyIsoCode"]]
    oppcurrency.columns = ["OpportunityId", "CurrencyIsoCode"]

    created = opps[["Id", "created"]]
    created.columns = ["Id", "CreatedDate"]

    # Melt/transpose the opps
    opps = pd.melt(
        opps,
        id_vars=["Id"],
        value_vars=[
            "ARR_Amount__c",
            "StageName",
            "CloseDate",
            "created",
            "NextStep"
        ],
        var_name="Field",
        value_name="NewValue",
    )

    opps = pd.merge(opps, created, how="left", on="Id")
    opps["CreatedDate"] = pd.to_datetime(opps["CreatedDate"])

    # 5. Determine Field Values
    fieldVals = pd.DataFrame(
        fieldHistory.groupby(["OpportunityId", "Field"])["CreatedDate"].nunique()
    ).reset_index()
    fieldVals = fieldVals[["OpportunityId", "Field"]]
    fieldVals.columns = ["Id", "Field"]

    # 6. Merge Opps and Field Values
    oppValsToAdd = pd.merge(opps, fieldVals, how="outer", indicator=True)
    oppValsToAdd = oppValsToAdd[oppValsToAdd["_merge"] == "left_only"]
    oppValsToAdd = oppValsToAdd.drop(columns="_merge")
    oppValsToAdd.rename(columns={"Id": "OpportunityId"}, inplace=True)

    # 7. Concatenate and Fill
    OppFieldHistory = pd.concat([oppValsToAdd, fieldHistory])
    OppFieldHistory = OppFieldHistory.fillna("None")
    
    # Ensure columns match for concatenation stability
    # The notebook code is a bit loose here, relying on matching columns by name.

    # 8. Handle First Old Value logic
    FirstOldValue = (
        OppFieldHistory.sort_values("CreatedDate")
        .groupby(["OpportunityId", "Field"])
        .first()
        .reset_index()
    )
    createdate = FirstOldValue.groupby(["OpportunityId"])["CreatedDate"].min()
    FirstOldValue["CreatedDate"] = FirstOldValue["OpportunityId"].apply(
        lambda x: createdate[x]
    )
    
    # Filter logic from notebook
    FirstOldValue = FirstOldValue[
        (((FirstOldValue["Field"] == "StageName") & (FirstOldValue["OldValue"] == "0")) | (FirstOldValue["OldValue"] != "0")) & (FirstOldValue["OldValue"] != "None")
    ]
    FirstOldValue["NewValue"] = FirstOldValue["OldValue"]
    FirstOldValue["OldValue"] = "None"
    
    OppFieldHistory = pd.concat([OppFieldHistory, FirstOldValue])
    OppFieldHistory.drop_duplicates(
        subset=["OpportunityId", "Field", "CreatedDate"], keep="first", inplace=True
    )
    OppFieldHistory.sort_values(by='CreatedDate', inplace=True)

    # 9. Pivot Table
    OppFieldHistory = OppFieldHistory.pivot_table(
        index=["OpportunityId", "CreatedDate"],
        values=["NewValue"],
        columns="Field",
        aggfunc="first",
    )
    OppFieldHistory = OppFieldHistory.droplevel(0, axis=1).reset_index()

    # 10. Forward Fill
    for col in OppFieldHistory.columns:
        OppFieldHistory = OppFieldHistory.sort_values(
            by=["OpportunityId", "CreatedDate"], ascending=[True, True]
        )
        OppFieldHistory[col] = (
            OppFieldHistory.groupby(OppFieldHistory["OpportunityId"])[col]
            .ffill()
            .infer_objects() # copy=False deprecated in newer pandas, infer_objects is safer default
        )

    OppFieldHistory = pd.merge(OppFieldHistory, oppcurrency, how="left", on="OpportunityId")
    OppFieldHistory = pd.merge(OppFieldHistory, currency, how="left", on="CurrencyIsoCode")

    # 11. Calculate Amounts
    valFields = ["ARR_Amount__c"]
    for col in valFields:
        if col in OppFieldHistory.columns:
            OppFieldHistory[col] = OppFieldHistory[col].replace("None", 0)
            OppFieldHistory[col] = OppFieldHistory[col].fillna(0)
            OppFieldHistory[col] = OppFieldHistory[col].astype("float32") # using float32 directly
            OppFieldHistory[col] = OppFieldHistory[col] / OppFieldHistory["Rate"]

    # 12. Final Polish
    # Filter Columns
    keep_cols = [
        "OpportunityId",
        "CreatedDate",
        "ARR_Amount__c",
        "CloseDate",
        "StageName",
        "NextStep"
    ]
    # Only keep columns that actually exist after pivot
    actual_keep = [c for c in keep_cols if c in OppFieldHistory.columns]
    OppFieldHistory = OppFieldHistory[actual_keep]

    if 'StageName' in OppFieldHistory.columns:
        OppFieldHistory['StageName'] = OppFieldHistory['StageName'].replace(lookup_dict)

    # 13. Merge with Current Opp State logic (Snapshotting)
    # Re-using the raw opps load for the 'opps' variable needed here
    # The notebook re-queries Opps at the end. We'll use our dummy_opportunities.csv
    opps_snapshot = pd.read_csv('dummy_opportunities.csv')
    opps_snapshot["CreatedDate"] = pd.to_datetime(opps_snapshot["CreatedDate"])
    # Not exporting the intermediate parquet 'OpportunityFieldHistoryOpps.parquet' unless needed

    # 14. Sort Key and ValidToDate
    OppFieldHistory['Opportunity History Sort Key'] = OppFieldHistory['CreatedDate'].dt.strftime('%Y%m%d%H%M%S') + OppFieldHistory['OpportunityId'].astype(str)
    
    OppFieldHistory = OppFieldHistory.sort_values(by=['OpportunityId', 'Opportunity History Sort Key'])
    OppFieldHistory['ValidToDate'] = OppFieldHistory.groupby('OpportunityId')['CreatedDate'].shift(-1)
    OppFieldHistory['Is Last Update'] = OppFieldHistory['ValidToDate'].isna().astype(int)

    # Merge stages
    if 'StageName' in OppFieldHistory.columns:
        OppFieldHistory = pd.merge(OppFieldHistory, stages, how='left', on='StageName')

    # Merge creation date
    OppFieldHistory = pd.merge(OppFieldHistory, opps_snapshot[['Id','CreatedDate']].rename(columns={'Id':'OpportunityId','CreatedDate':'Opportunity Created Date'}), how='left', on='OpportunityId')

    if 'StageName' in OppFieldHistory.columns:
        OppFieldHistory['Stage IsLost'] = OppFieldHistory['StageName'] == "Closed Lost"
        OppFieldHistory['Stage IsDQ'] = OppFieldHistory['StageName'] == "Disqualified"
    
    if 'ARR_Amount__c' in OppFieldHistory.columns:
        OppFieldHistory['Amount USD'] = OppFieldHistory['ARR_Amount__c']
        OppFieldHistory = OppFieldHistory.rename(columns={'ARR_Amount__c':'AmountUSD'})

    # Filter out empty amounts
    if 'AmountUSD' in OppFieldHistory.columns:
         OppFieldHistory = OppFieldHistory.groupby('OpportunityId').filter(lambda x: not all(x['AmountUSD'] == 0))

    print("Initial processing complete. DataFrame head:")
    print(OppFieldHistory.head())

    # Export
    print("Exporting to OpportunityFieldHistory_PoC.parquet...")
    OppFieldHistory.to_parquet('OpportunityFieldHistory_PoC.parquet', index=False)
    print("Done.")

if __name__ == "__main__":
    run_pipeline_history_poc()
