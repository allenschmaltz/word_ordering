package org.maltparser.core.lw.parser;


import org.maltparser.core.exception.MaltChainedException;
import org.maltparser.core.feature.FeatureVector;
import org.maltparser.core.feature.value.FeatureValue;
import org.maltparser.core.feature.value.MultipleFeatureValue;
import org.maltparser.core.feature.value.SingleFeatureValue;
import org.maltparser.ml.lib.FeatureList;
import org.maltparser.ml.lib.FeatureMap;
import org.maltparser.ml.lib.MaltLibModel;
import org.maltparser.ml.lib.LibException;
import org.maltparser.parser.history.action.SingleDecision;

/**
* A lightweight version of org.maltparser.ml.lib.{Lib,LibLinear,LibSvm} and can only predict the next transition.
* 
* @author Johan Hall
*/
public class LWClassifier {
	private final FeatureMap featureMap;
	private final boolean excludeNullValues;
	private final MaltLibModel model;

	public LWClassifier(McoModel mcoModel, String prefixFileName, boolean _excludeNullValues)  {
		this.model = (MaltLibModel)mcoModel.getMcoEntryObject(prefixFileName+".moo");
		this.featureMap = (FeatureMap)mcoModel.getMcoEntryObject(prefixFileName+".map");
		this.excludeNullValues = _excludeNullValues;
	}
	
	public boolean predict(FeatureVector featureVector, SingleDecision decision, boolean one_prediction) throws MaltChainedException {
		final FeatureList featureList = new FeatureList();
		final int size = featureVector.size();
		for (int i = 1; i <= size; i++) {
			final FeatureValue featureValue = featureVector.getFeatureValue(i-1);	
			if (featureValue != null && !(excludeNullValues == true && featureValue.isNullValue())) {
				if (!featureValue.isMultiple()) {
					SingleFeatureValue singleFeatureValue = (SingleFeatureValue)featureValue;
					final int index = featureMap.getIndex(i, singleFeatureValue.getIndexCode());
					if (index != -1 && singleFeatureValue.getValue() != 0) {
						featureList.add(index,singleFeatureValue.getValue());
					}
				} 
				else { 
					for (Integer value : ((MultipleFeatureValue)featureValue).getCodes()) {
						final int v = featureMap.getIndex(i, value);
						if (v != -1) {
							featureList.add(v,1);
						}
					}
				} 
			}
		}
		try {
			if (one_prediction) {
				decision.getKBestList().add(model.predict_one(featureList.toArray()));
			} else {
				decision.getKBestList().addList(model.predict(featureList.toArray()));
			}
		} catch (OutOfMemoryError e) {
			throw new LibException("Out of memory. Please increase the Java heap size (-Xmx<size>). ", e);
		}
		return true;
	}
}
