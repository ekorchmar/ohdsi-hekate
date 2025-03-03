-- Drug strength entries that specify invalid units
select
    ds.drug_concept_id,
    au.concept_id is NULL as amount_unit,
    nu.concept_id is NULL as numerator_unit,
    du.concept_id is NULL as denominator_unit,
    coalesce(au.concept_id, nu.concept_id, du.concept_id) as concept_id,
    coalesce(au.concept_name, nu.concept_name, du.concept_name) as concept_name
from drug_strength as ds
inner join concept as d
    on
        ds.drug_concept_id = d.concept_id
        and ds.invalid_reason is NULL
        and d.standard_concept = 'S'
left join
    concept as au
    on
        ds.amount_unit_concept_id = au.concept_id
        and au.invalid_reason is not NULL
left join
    concept as nu
    on
        ds.numerator_unit_concept_id = nu.concept_id
        and nu.invalid_reason is not NULL
left join
    concept as du
    on
        ds.denominator_unit_concept_id = du.concept_id
        and du.invalid_reason is not NULL
where
    coalesce(au.concept_id, nu.concept_id, du.concept_id) is not NULL
